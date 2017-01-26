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
from torchy.pipeline import writer

class Controller():
    """ 
    Controller class that does the dask book keeping and 
    transforms request to a dask graph
    """
    def __init__(self):
        self.current_targets = []
        self.current_dsk = []
        self.current_requests = None
        self.classifier_type = "VigraRF"
        self.output_file = None

    def clear_dsk(self):
        self.current_targets = []
        self.current_dsk = []
        self.current_requests = None
 
    def process_requests(self, json_base_name):
        self.clear_dsk()
        requests = fetch_new_requests(json_base_name)
        dsk = {"RF":self.classifier_type}
        self.current_requests = requests

        if self.output_file is None:
            if_set = set([r["data_filename"] for r in requests])
            print(if_set)
            if len(if_set) > 1:
                raise NotImplementedError("Sorry: we can only handle one lane (single input file)")

            self.output_file =  "results/full_prediction.h5"
            with h5py.File(self.output_file, "w") as f:
                with h5py.File(if_set.pop(), "r") as raw_f:
                    # TODO: replace the RF location by json defined file path
                    rf_file = "datasets/hackathon_flyem_forest.h5"
                    n_classes = learning.count_labels(learning.get_classifier(self.classifier_type, filename=rf_file))
                    prediction_shape = [1, n_classes] + list(raw_f["volume/data"].shape)[:-1]
                f.create_dataset("data", prediction_shape)
            print ("created output file %s"%self.output_file)

        for r in requests:
            rid = r["req_id"]
            r["output_file_name"] = self.output_file
            dsk["request-%i"%rid] = r
            dsk["RF-File-%i"%rid] = "datasets/hackathon_flyem_forest.h5"# should be r["classifier_filename"] when we get the correct json files
            dsk["computed-features-%i"%rid] = (vigra_filters.dummy_feature_prediciton, "request-%i"%rid)
            dsk["predicted-image-%i"%rid] = (learning.image_prediction, "RF", "RF-File-%i"%rid, 'computed-features-%i'%rid, "request-%i"%rid)
            # dsk["write-%i"%rid] = (writer.write_output, "predicted-image-%i"%rid, "request-%i"%rid)
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
            data["halo_size"] = 20
            cutout_to_slice(data)
            requests.append(data)
    return requests

def m(xyz):
    # limit the halo size if halo would surpass the boundary
    return max(0,xyz)

def cutout_to_slice(request):
    """
    conversion function from request format ROI to list of python slices
    that follow the data format convention (batch, channel, x, y, z)
    halo_size: number of pixels that are added to every boundary
    """

    c = request["cutout"][0]
    hs = request["halo_size"]

    if len(c)/2 == 2:
        request["roi_with_halo"] = (slice(None), slice(None), 
                slice(m(c['xmin']-hs),c['xmax']+hs),
                slice(m(c['ymin']-hs),c['ymax']+hs), 0)
        request["roi_output"] = (slice(None), slice(None), 
                slice(m(c['xmin']),c['xmax']),
                slice(m(c['ymin']),c['ymax']))

    if len(c)/2 == 3:
        request["roi_with_halo"] = (slice(m(c['xmin']-hs),c['xmax']+hs),
                slice(m(c['ymin']-hs),c['ymax']+hs),
                slice(m(c['zmin']-hs),c['zmax']+hs), 0)
        request["roi_output"] = (slice(None), slice(None), 
                slice(m(c['xmin']),c['xmax']),
                slice(m(c['ymin']),c['ymax']),
                slice(m(c['zmin']),c['zmax']))


