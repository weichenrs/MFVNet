import os 

class Path(object):
    @staticmethod
    def db_root_dir(dataset, scale):
        if dataset == 'potsdam':
            datapath = '../data/potsdam/'
            # return '../data/potsdam/512'
            # return '../data/potsdam/768'
            # return '../data/potsdam/1024'
            # return '../data/potsdam/1280'
        elif dataset == 'wfv':
            datapath = '../data/wfv/'
            # return '../data/wfv/512'
            # return '../data/wfv/768'
            # return '../data/wfv/1024'
            # return '../data/wfv/1280'
        elif dataset == 'gid':
            datapath = '../data/gid/'
            # return '../data/gid/512'
            # return '../data/gid/768'
            # return '../data/gid/1024'
            # return '../data/gid/1280'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
        
        return os.path.join(datapath, scale)
