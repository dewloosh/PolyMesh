# -*- coding: utf-8 -*-
from typing import Iterable
import numpy as np
import awkward as ak

from dewloosh.core.wrapping import Wrapper


class AkWrapper(Wrapper):
    """
    A wrapper for Awkward objects. This is for example the master class
    of all finite element classes.
    
    Although is is not likely, it can be used directly and enjoy the input/output
    capabilities of the class.
    
    """
    
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
        """
        Returns the wrapped Awkward object.
        """
        return self._wrapped

    def to_numpy(self, key):
        """
        Returns a data with the specified key as a numpy array, if possible.
        """
        return self._wrapped[key].to_numpy()
    
    def to_pandas(self, *args, fields: Iterable[str] = None, **kwargs):
        """
        Returns the data contained within the database as a DataFrame.

        Parameters
        ----------
        path_pd : str
            File path for point-related data.

        path_cd : str
            File path for cell-related data.

        fields : Iterable[str], Optional
            Valid field names to include in the parquet files.

        """
        akdb = self.to_ak(*args, fields=fields, **kwargs)
        return ak.to_pandas(akdb)

    def to_parquet(self, path: str, *args, fields: Iterable[str] = None, **kwargs):
        """
        Saves the data contained within the database to a parquet file.

        Parameters
        ----------
        path : str
            Path of the file being created.

        fields : Iterable[str], Optional
            Valid field names to include in the parquet files.

        """
        akdb = self.to_ak(*args, fields=fields, **kwargs)
        ak.to_parquet(akdb, path)

    def to_ak(self, *args, fields: Iterable[str] = None, **kwargs):
        """
        Returns the data contained within the mesh as an Awkward array.

        Parameters
        ----------
        fields : Iterable[str], Optional
            Valid field names to include in the returned objects.
        """
        ldb = self.to_list(*args, fields=fields, **kwargs)
        return ak.from_iter(ldb)

    def to_list(self, *args, fields: Iterable[str] = None, **kwargs) -> list:
        """
        Returns data of the object as a lists. Unless specified by 'fields', 
        all fields are returned.

        Parameters
        ----------
        fields : Iterable[str], Optional
            A list of keys that might identify data in a database. Default is None.

        """
        db = self.db
        res = None
        if fields is not None:
            db_ = {}
            for f in fields:
                if f in db.fields:
                    db_[f] = db[f]
            res = AkWrapper(fields=db_).to_list()
        else:
            res = db.to_list()
        return res
    
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

    