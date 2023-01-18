import multiprocessing

class ParallelProcess:
    def __init__(self):
        self.shared_variable = multiprocessing.Value('i', 0)

    def function_1(self):
        # code for function 1
        with self.shared_variable.get_lock():
            self.shared_variable.value += 1

    def function_2(self):
        # code for function 2
        with self.shared_variable.get_lock():
            self.shared_variable.value -= 1

    def run(self):
        p1 = multiprocessing.Process(target=self.function_1)
        p2 = multiprocessing.Process(target=self.function_2)
        p1.start()
        p2.start()
        p1.join()
        p2.join()

if __name__ == '__main__':
    parallel_process = ParallelProcess()
    parallel_process.run()
    print(parallel_process.shared_variable.value)