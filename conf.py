import os.path


class ModelConf(object):
    def __init__(self,file):
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, item):
        # print(item)
        if not self.contain(item):
            print('parameter '+item+' is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self,key):
        # print(key)
        return key in self.config

    def read_configuration(self,file):
        if not os.path.exists(file):
            print('config file is not found!')
            raise IOError
        with open(file) as f:
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        key,value=line.strip().split('=')
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d' % ind)


class OptionConf(object):
    def __init__(self, config_file=None, config=None):
        if config_file:
            self.config = {}
            self.read_conf(config_file)
        else:
            self.config = config

    def __getitem__(self, item):
        return self.config[item]

    def __iter__(self):
        return iter(self.config)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, item):
        return item in self.config

    def get(self, item, default=None):
        if item in self.config:
            return self.config[item]
        return default

    def keys(self):
        return self.config.keys()

    def values(self):
        return self.config.values()

    def contains(self, key_name):
        return key_name in self.config

    def read_conf(self, config_file):
        if not os.path.exists(config_file):
            print('config_file is not found!')
            raise IOError
        with open(config_file) as f:
            for ind, line in enumerate(f):
                if line.strip() != '':
                    try:
                        key, value = line.strip().split('=')
                        self.config[key] = value
                    except ValueError:
                        print('config file %s at line %d has format problem.' % (config_file, ind))

    def options(self, opt_name):
        ops = {}
        for key, value in self.config.items():
            prefix = opt_name + ':'
            if key.startswith(prefix):
                ops[key[len(prefix):]] = value
        return ops

    def save_conf(self, config_file):
        with open(config_file, 'w') as f:
            for key, value in self.config.items():
                f.write('%s=%s\n' % (key, value))


