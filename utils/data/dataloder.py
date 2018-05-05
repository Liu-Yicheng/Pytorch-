import torch
import torch.multiprocessing as multiprocessing
from .sampler import SequentialSampler, RandomSampler, BatchSampler
import collections
import re
import sys
import traceback
import threading
from torch._six import string_classes


if sys.version_info[0] == 2:
    "get the main version of the python,so you can run this program on different versions"
    import Queue as queue
else:
    import queue


_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    torch.set_num_threads(1) #设置为一个线程
    while True:
        r = index_queue.get()#传入的index_queue是一个简单队列,先进先出
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except Exception:
            # is_set判断内部信号Flag标志状态，Event事件实现通信机制：全局定义了一个“Flag”（默认为False），
            # 若Flag信号被clear为False，则执行event.wait方法时会阻塞；若Flag信号被set为True，则执行event.wait方法时便不阻塞。
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])#返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
            storage = batch[0].storage()._new_shared(numel)#在share_memory中创建一个numel大小的,类型与batch[0].storage()一样的storage
            out = batch[0].new(storage)#猜想应该是给srotage做初始化
        return torch.stack(batch, 0, out=out)#将batch里的stack成[batch_size, batch[0]的维度0, .1, .2, ...]
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

'''pin_memory_batch函数不是定义在DataLoader类或DataLoaderIter类中。该函数主要是对batch中的Tensor执行batch.pin_memory()操作
，这里的很多条件语句只是用来判断batch的类型，假如batch是一个列表，列表中的每个值是Tensor，那么就会执行 elif isinstance(batch, collections.Sequence):\
这个条件，从而遍历该列表中的每个Tensor，然后执行第一个条件语句的内容： return batch.pin_memory()
'''
def pin_memory_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        '''self.sample_iter = iter(self.batch_sampler)，得到的self.sample_iter可以通过next(self.sample_iter)来获取batch size个数据的index'''
        self.sample_iter = iter(self.batch_sampler)

        '''if self.num_workers语句是针对多进程或单进程的情况进行初始化，如果不是设置为多进程读取数据，那么就不需
            要这些初始化操作，后面会介绍单进程数据读取。'''
        if self.num_workers > 0:

            '''在if语句中通过multiprocessing.SimpleQueue()
            类创建了一个简单的队列对象。multiprocessing.Process
            类就是构造进程的类，这里根据设定的进程数来启动，然后赋值给self.workers'''
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            '''self.rcvd_idx表示读取到的一个batch数据的index，初始化为0，该值在迭代读取数据的时候会用到'''
            self.rcvd_idx = 0
            self.reorder_dict = {}

            '''for循环就通过调用start方法依次启动self.workers中的进程'''
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            '''self.pin_memory的判断语句，该判断语句内部主要是实现了多线程操作,self.pin_memory的含义在前面已经介绍过了，
                当为True的时候，就会把数据拷到CUDA'''
            if self.pin_memory:
                in_data = self.data_queue
                '''self.data_queue = queue.Queue()是通过Python的queue模块初始化得到一个先进先出的队列
                （queue模块也可以初始化得到先进后出的队列，需要用queue.LifoQueue()初始化），queue模块主要应用在多线程读取数据中'''
                self.data_queue = queue.Queue()

                ''' 在threading.Thread的args参数中，第一个参数in_data就是一个进程的数据，
                    一个进程中不同线程的数据也是通过队列来维护的，这里采用的是Python的queue模块来初始化得到一个队列：queue.Queue()。
                    初始化结束后，就会调用__next__方法，接下来介绍。总的来说，如果设置为多进程读取数据，那么就会采用队列的方式来读，
                    如果不是采用多进程来读取数据，那就采用普通方式来读。'''
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True #因子进程设置了daemon属性，主进程结束，它们就随着结束了。
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):

        '''用来处理self.num_workers等于0的情况，也就是不采用多进程进行数据读取'''
        if self.num_workers == 0:  # same-process loading

            '''先通过indices = next(self.sample_iter)获取长度为batch size的列表：indices
                这个列表的每个值表示一个batch中每个数据的index，每执行一次next操作都会读取一批长度为batch size的indices列表
            '''
            indices = next(self.sample_iter)  # may raise StopIteration

            '''self.collate_fn函数将batch size个tuple（每个tuple长度为2，其中第一个值是数据，Tensor类型，
               第二个值是标签，int类型）封装成一个list，这个list长度为2，两个值都是Tensor，一个是batch size个数据组成的FloatTensor
，              另一个是batch size个标签组成的LongTensor。
                所以简单讲self.collate_fn函数就是将batch size个分散的Tensor封装成一个Tensor'''
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:

                '''batch = pin_memory_batch(batch)中pin_memory_batch函数的作用就是将输入batch的每个Tensor都拷贝到CUDA中，该函数后面会详细介绍'''
                batch = pin_memory_batch(batch)
            return batch

        # check if the next sample has already been generated
        '''第二个if语句是判断当前想要读取的batch的index(self.rcvd_idx)是否之前已经读出来过(已读出来的index和batch数据保存
           在self.reorder_dict字典，可以结合最后的while语句一起看，因为self.reorder_dict字典的更新是在最后的while语句中）
           如果之前已经读取过了，就根据这个index从reorder_dict字典中弹出对应的数据。最后返回batch数据的时候是 
           return self._process_next_batch(batch)，该方法后面会详细介绍。主要做是获取下一个batch的数据index信息。
        '''
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)


        '''第三个if语句，self.batches_outstanding的值在前面初始中调用self._put_indices()方法时修改了，
           所以假设你的进程数self.num_workers设置为3.那么这里self.batches_outstanding就是3*2=6。这个的值
           其实代表队列中保存的batch的个数，每当从data_queue.get()时，值减少1,每当调用_put_indices时，值加一
           当队列中的数据都被读取完了，进程就停止工作'''
        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        '''最后的while循环就是真正用来从队列中读取数据的操作，最主要的就是idx, batch = self._get_batch()，
           通过调用_get_batch()方法来读取，后面有介绍，简单讲就是调用了队列的get方法得到下一个batch的数据，
           得到的batch一般是长度为2的列表，列表的两个值都是Tensor，分别表示数据（是一个batch的）和标签'''
        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1

            '''_get_batch()方法除了返回batch数据外，还得到另一个输出：idx，这个输出表示batch的index，
               这个if idx != self.rcvd_idx条件语句表示如果你读取到的batch的index不等于当前想要的index:selg,rcvd_idx，
               那么就将读取到的数据保存在字典self.reorder_dict中：self.reorder_dict[idx] = batch，然后继续读取数据，
               直到读取到的数据的index等于self.rcvd_idx。个人感觉这种情况应是队列里的次序被打乱的时候'''
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    '''首先对self.rcvd_idx进行加一，也就是更新下下一个要读取的batch数据的index。然后调用_put_indices()方法获取下一个batch的每个数据的index。'''
    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False):
        self.dataset = dataset         #用户自定义的数据类
        self.batch_size = batch_size   #用户定义的batch_size大小
        self.num_workers = num_workers #用户定义的操作数据的进程数
        self.collate_fn = collate_fn   #与储存张量到内存有关
        self.pin_memory = pin_memory   #是否将处理的数据放入CUDA中
        self.drop_last = drop_last     #在对数据集形成batch时，当最后一个batch的数量不足batchsize时，是否舍弃

        '''batch_size, shuffle, sampler是用来构造默认的batch_sample
           当用户自己定义了batch_sample时，就不需要这些参数了。
           batch_sample实际上是一个迭代器，迭代对象是一个batch的图片idx。
           '''
        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)



    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)