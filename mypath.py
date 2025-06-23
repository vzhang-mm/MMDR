import platform
from opts import parse_opts
args = parse_opts()

class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'avec2014':#使用图像帧
            root_dir = 'F:/MDDR/ZMT/root/avec2014/'#2014
            return root_dir
        elif database == 'avec2013':#使用图像帧
            root_dir = 'D:/Desktop/MDDR/ZMT/root/avec2013/'#2014
            return root_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return ''
