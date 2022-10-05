def soft_update(target, source, polyak):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * polyak + param.data * (1.0 - polyak))


class ArgsHolder(object):
    def __init__(self, args):
        self.keys = list()
        for key, value in dict(args).items():
            setattr(self, key, value)
            self.keys.append(key)

    def merge(self, args):
        for key, value in dict(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)
                self.keys.append(key)
            else:
                raise KeyError(f'Config attribute {key} already exists')

    def to_json(self):
        json = dict()
        for key in self.keys:
            json[key] = getattr(self, key)
        return json
