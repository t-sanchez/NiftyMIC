##
# \file outlier_rejector.py
# \brief      Class to identify and reject outliers.
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       Jan 2019
#

import os
import csv
import scipy
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import pysitk.python_helper as ph
import pysitk.simple_itk_helper as sitkh
import niftymic.validation.residual_evaluator as re


##
# Class to identify and reject outliers
# \date       2019-01-28 19:24:52+0100
#
class OutlierRejector(object):

    def __init__(self,
                 stacks,
                 reference,
                 threshold,
                 cycle,
                 use_slice_masks=False,
                 use_reference_mask=True,
                 measure="NCC",
                 output_dir="srr/recon_subject_space",
                 verbose=True,
                 ):

        self._stacks = stacks
        self._reference = reference
        self._threshold = threshold
        self._cycle = cycle
        self._measure = measure
        self._use_slice_masks = use_slice_masks
        self._use_reference_mask = use_reference_mask
        self._output_dir = output_dir
        self._verbose = verbose

    def get_stacks(self):
        return self._stacks

    def run(self):
        residual_evaluator = re.ResidualEvaluator(
            stacks=self._stacks,
            reference=self._reference,
            use_slice_masks=self._use_slice_masks,
            use_reference_mask=self._use_reference_mask,
            verbose=False,
            measures=[self._measure],
        )
        residual_evaluator.compute_slice_projections()
        residual_evaluator.evaluate_slice_similarities()
        slice_sim = residual_evaluator.get_slice_similarities()
        # residual_evaluator.show_slice_similarities(
        #     threshold=self._threshold,
        #     measures=[self._measure],
        #     directory="/tmp/spina/figs%s" % self._print_prefix[0:7],
        # )

        # Creating the path towards the csv file where the slice rejection data will be stored
        output_dir_path = "/srr/recon_subject_space"
        output_file_path = os.path.join(output_dir_path, "srr_rejectedSlices.csv")

        remove_stacks = []
        for i, stack in enumerate(self._stacks):
            nda_sim = np.nan_to_num(
                slice_sim[stack.get_filename()][self._measure])
            indices = np.where(nda_sim < self._threshold)[0]
            slices = stack.get_slices()

            # only those indices that match the available slice numbers
            rejections = [
                j for j in [s.get_slice_number() for s in slices]
                if j in indices
            ]

            for slice in slices:
                if slice.get_slice_number() in rejections:
                    stack.delete_slice(slice)

            if self._verbose:
                txt = "Stack %d/%d (%s): Slice rejections %d/%d [%s]" % (
                    i + 1,
                    len(self._stacks),
                    stack.get_filename(),
                    len(stack.get_deleted_slice_numbers()),
                    stack.sitk.GetSize()[-1],
                    ph.convert_numbers_to_hyphenated_ranges(
                        stack.get_deleted_slice_numbers()),
                )
                if len(rejections) > 0:
                    res_values = nda_sim[rejections]
                    txt += " | Latest rejections: " \
                        "%d [%s] (%s < %g): %s" % (
                            len(rejections),
                            ph.convert_numbers_to_hyphenated_ranges(
                                rejections),
                            self._measure,
                            self._threshold,
                            np.round(res_values, 2).tolist(),
                        )
                ph.print_info(txt)

            # Get list of all slices even those which were previously deleted
            stack_list = [s.get_slice_number() for s in stack.get_slices()] + stack.get_deleted_slice_numbers()
            stack_list.sort()

            stack_filename = stack.get_filename()
            subject_name = stack_filename.split("_")[0]

            # Loop over all slices that exist and/or existed within a stack 
            stack_state =[]
            for slice in stack_list:
                slice_state = [] # (subject_name, stack_filename, mask_filename,
                                # slice, stack, cycle, measure, threshold,
                                # NCC_value, just_rejected, rejected)

                slice_state.append(subject_name)
                slice_state.append(stack_filename + ".nii.gz")
                slice_state.append(stack_filename + "_mask.nii.gz")
                slice_state.append(slice)
                slice_state.append(i+1)
                slice_state.append(self._cycle)
                slice_state.append(self._measure)
                slice_state.append(self._threshold)
                slice_state.append(nda_sim[slice] if slice in [s.get_slice_number() for s in slices] else np.nan)
                slice_state.append(int(slice in rejections))
                slice_state.append(int(slice in stack.get_deleted_slice_numbers()))

                stack_state.append(slice_state)

            # Log stack where all slices were rejected
            if stack.get_number_of_slices() == 0:
                remove_stacks.append(stack)


            if os.path.exists(output_file_path):
                if (os.stat(output_file_path).st_size == 0):
                    with open(output_file_path, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(["subject_name", "stack_filename",
                                        "mask_filename", "slice", "stack",
                                        "cycle", "measure", "threshold",
                                        "NCC_value", "just_rejected", "rejected"])
                else:
                    with open(output_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerows(stack_state)
            else:
                with open(output_file_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["subject_name", "stack_filename",
                                    "mask_filename", "slice", "stack",
                                    "cycle", "measure", "threshold",
                                    "NCC_value", "just_rejected", "rejected"])
                    writer.writerows(stack_state)


        # Remove stacks where all slices where rejected
        for stack in remove_stacks:
            self._stacks.remove(stack)
            if self._verbose:
                ph.print_info("Stack '%s' removed entirely." %
                              stack.get_filename())
