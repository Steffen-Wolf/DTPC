from __future__ import print_function

from torchy.utils import timeit
from torchy.pipeline import dask_controller
from torchy.learning import learning

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hold tight for pure cpu, gpu ilastik goodness')
    parser.add_argument('--ndim', dest='ndim', type=int, default=3)
    parser.add_argument('--num_workers_per_computer', type=int, default=3)
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--max_edge_length', type=int, default=200)
    parser.add_argument('--requests', type=str, default="/export/home/nrahaman/Python/Repositories/"
                                                        "TorchyMcDaskface/datasets/"
                                                        "sample_c0_req0.json")
    parser.add_argument('--halo_size', type=int, default=7)
    parser.add_argument('--classifier_type', type=str, default="RandomForest")
    parser.add_argument('--rf_source', type=str, default="from_request")

    options = parser.parse_args()
    P = dask_controller.Controller(options)
    P.build_feature_computer_pool(num_computers=options.num_workers,
                                  ndim=options.ndim,
                                  num_workers_per_computer=options.num_workers_per_computer,
                                  device=options.device)

    P.process_requests(options.requests)

    with timeit() as time_info:
        output = P.get_results()
    print(output)
    print("Time Elapsed: {}".format(time_info.elapsed_time))

