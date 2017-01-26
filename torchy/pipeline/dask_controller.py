"""This file contains the dask pipeline creation and request handling API"""

from __future__ import print_function
import numpy as np
from dask.threaded import get

import vigra
import json

import h5py
import os
from glob import glob

from torchy.learning import learning
from torchy.filters import filters
from torchy.filters import vigra_filters

class Controller():
    """ 
    Controller class that does the dask book keeping and 
    transforms request to a dask graph
    """
    def __init__(self):
        self.current_targets = []
        self.current_dsk = []
        self.current_requests = None
        self.random_forest_type = "VigraRF"

    def clear_dsk(self):
        self.current_targets = []
        self.current_dsk = []
        self.current_requests = None
 
    def process_requests(self, json_base_name):
        self.clear_dsk()
        requests = fetch_new_requests(json_base_name)
        dsk = {"RF":self.random_forest_type}
        self.current_requests = requests

        for r in requests:
            rid = r["req_id"]
            dsk["request-%i"%rid] = r
            dsk["RF-File-%i"%rid] = "datasets/hackathon_flyem_forest.h5"# should be r["classifier_filename"] when we get the correct json files
            dsk["computed-features-%i"%rid] = (vigra_filters.dummy_feature_prediciton, "request-%i"%rid)
            dsk["predicted-image-%i"%rid] = (learning.image_prediction, "RF", "RF-File-%i"%rid, 'computed-features-%i'%rid)
            self.current_targets.append("predicted-image-%i"%rid)
        self.current_dsk = dsk

    def get_results(self):
        return get(self.current_dsk, self.current_targets)

    def get_request(self):
        return self.current_requests


def fetch_new_requests(base_name):
    requests = []
    for json_file_name in glob(base_name):
        with open(json_file_name) as data_file:    
            data = json.load(data_file)
            data["roi_with_halo"] = cutout_to_slice(data["cutout"][0])
            requests.append(data)
    return requests

def cutout_to_slice(cutout, halo_size=0):
    """
    conversion function from request format ROI to list of python slices
    that follow the data format convention (batch, channel, x, y, z)
    halo_size: number of pixels that are added to every boundary
    """

    # rename stuff to make the following more readable :)
    c = cutout
    hs = halo_size

    if len(c)/2 == 2:
        return (slice(None), slice(None), 
                slice(c['xmin']-hs,c['xmax']+hs),
                slice(c['ymin']-hs,c['ymax']+hs), 0)
    if len(c)/2 == 3:
        return (slice(c['xmin']-hs,c['xmax']+hs),
                slice(c['ymin']-hs,c['ymax']+hs),
                slice(c['zmin']-hs,c['zmax']+hs), 0)


