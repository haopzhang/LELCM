import time


class Log:
    def __init__(self, task):
        self.name = "./logs/"+time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))+"_"+task+".txt"
        with open(self.name, 'a+', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))+"    "+"task"+'\n')

    def log_print(self, message):
        with open(self.name, 'a+', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))+"    "+message+'\n')
            print(message)