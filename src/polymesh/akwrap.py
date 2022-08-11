# -*- coding: utf-8 -*-
import numpy as np
import awkward as ak

from ..core import Wrapper


class AkWrapper(Wrapper):
    
    _attr_map_ = {}

    def __init__(self, *args, wrap=None, fields=None, **kwargs):
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)
        if wrap is None and (len(kwargs) + len(fields)) > 0:
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    fields[k] = v
            if len(fields) > 0:
                wrap = ak.zip(fields, depth_limit=1)
        if len(kwargs) > 0:
            [kwargs.pop(k, None) for k in fields.keys()]
        super().__init__(*args, wrap=wrap, **kwargs)

    @property
    def db(self):
        return self._wrapped

    def to_numpy(self, key):
        return self._wrapped[key].to_numpy()
    
    def __len__(self):
        return len(self._wrapped)
    
    def __hasattr__(self, attr):
        if attr in self.__class__._attr_map_:
            attr = self.__class__._attr_map_[attr]            
        return any([attr in self.__dict__, 
                    attr in self._wrapped.__dict__])

    def __getattr__(self, attr):
        if attr in self.__class__._attr_map_:
            attr = self.__class__._attr_map_[attr]
        if attr in self.__dict__:
            return getattr(self, attr)
        try:
            return getattr(self._wrapped, attr)
        except Exception:
            raise AttributeError("'{}' object has no "\
                + "attribute called {}".format(self.__class__.__name__, attr))

    