#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def run_proc(name):
    import os
    print('Run child process %s (%s)...' % (name, os.getpid()))

def long_time_task(name):
    import random
    import time
    import os
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

# 写数据进程执行的代码:
def write(q):
    import os, time, random
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    import os
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

'''
if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p1 = Process(target=run_proc, args=('test1',))
    p2 = Process(target=run_proc, args=('test2',))
    p3 = Process(target=run_proc, args=('test3',))
    p4 = Process(target=run_proc, args=('test4',))
    print('Child process will start.')
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p4.close()
    print('Child process end.')
'''

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')