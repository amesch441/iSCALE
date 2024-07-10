from valis.registration import *


class ValisFixed(Valis):

    @valtils.deprecated_args(max_non_rigid_registartion_dim_px="max_non_rigid_registration_dim_px")
    def register_micro(self,  brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                 brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                 if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                 max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
                 non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
                 non_rigid_reg_params=DEFAULT_NON_RIGID_KWARGS,
                 reference_img_f=None, align_to_reference=False, mask=None, tile_wh=DEFAULT_NR_TILE_WH):
        """Improve alingment of microfeatures by performing second non-rigid registration on larger images

        Caclculates additional non-rigid deformations using a larger image

        Parameters
        ----------
        brightfield_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process brightfield images to make
            them look as similar as possible.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process immunofluorescent images
            to make them look as similar as possible.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        max_non_rigid_registration_dim_px : int, optional
             Maximum width or height of images used for non-rigid registration.
             If None, then the full sized image will be used. However, this
             may take quite some time to complete.

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference, and
            images will be aligned towards it. If provided, images will be
            aligned to this reference.

        align_to_reference : bool, optional
            If `False`, images will be non-rigidly aligned serially towards the
            reference image. If `True`, images will be non-rigidly aligned
            directly to the reference image. If `reference_img_f` is None,
            then the reference image will be the one in the middle of the stack.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images. See
            the `non_rigid_registrars` module for a desciption of available
            methods. If a desired non-rigid registration method is not available,
            one can be implemented by subclassing.NonRigidRegistrar.

        non_rigid_reg_params: dictionary, optional
            Dictionary containing key, value pairs to be used to initialize
            `non_rigid_registrar_cls`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

        """
        ref_slide = self.get_ref_slide()
        if mask is None:
            if ref_slide.non_rigid_reg_mask is not None:
                mask = ref_slide.non_rigid_reg_mask.copy()

        nr_reg_src, max_img_dim, non_rigid_reg_mask, full_out_shape_rc, mask_bbox_xywh = \
            self.prep_images_for_large_non_rigid_registration(max_img_dim=max_non_rigid_registration_dim_px,
                                                                brightfield_processing_cls=brightfield_processing_cls,
                                                                brightfield_processing_kwargs=brightfield_processing_kwargs,
                                                                if_processing_cls=if_processing_cls,
                                                                if_processing_kwargs=if_processing_kwargs,
                                                                updating_non_rigid=True,
                                                                mask=mask)

        img0 = nr_reg_src[serial_non_rigid.IMG_LIST_KEY][0]
        img_specific_args = None
        write_dxdy = False

        self._non_rigid_bbox = mask_bbox_xywh
        self._full_displacement_shape_rc = full_out_shape_rc



        if isinstance(img0, pyvips.Image):

            # Have determined that these images will be too big
            msg = (f"Registration would more than {TILER_THRESH_GB} GB if all images opened in memory. "
                    f"Will use NonRigidTileRegistrar to register cooresponding tiles to reduce memory consumption, "
                    f"but this method is experimental")

            valtils.print_warning(msg)

            write_dxdy = True
            img_specific_args = {}
            for slide_obj in self.slide_dict.values():

                # Add registration parameters
                tiled_non_rigid_reg_params = {}
                tiled_non_rigid_reg_params[non_rigid_registrars.NR_CLS_KEY] = non_rigid_registrar_cls
                tiled_non_rigid_reg_params[non_rigid_registrars.NR_STATS_KEY] = self.target_processing_stats
                tiled_non_rigid_reg_params[non_rigid_registrars.NR_TILE_WH_KEY] = tile_wh

                if slide_obj.is_rgb:
                    processing_cls = brightfield_processing_cls
                    processing_args = brightfield_processing_kwargs
                else:
                    processing_cls = if_processing_cls
                    processing_args = if_processing_kwargs

                tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_CLASS_KEY] = processing_cls
                tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_KW_KEY] = processing_args

                img_specific_args[slide_obj.src_f] = tiled_non_rigid_reg_params

            non_rigid_registrar_cls = non_rigid_registrars.NonRigidTileRegistrar

        print("\n==== Performing microregistration\n")
        non_rigid_registrar = serial_non_rigid.register_images(src=nr_reg_src,
                                                               non_rigid_reg_class=non_rigid_registrar_cls,
                                                               non_rigid_reg_params=non_rigid_reg_params,
                                                               reference_img_f=reference_img_f,
                                                               mask=non_rigid_reg_mask,
                                                               align_to_reference=align_to_reference,
                                                               name=self.name,
                                                               img_params=img_specific_args
                                                               )

        pathlib.Path(self.micro_reg_dir).mkdir(exist_ok=True, parents=True)
        out_shape = full_out_shape_rc
        n_digits = len(str(self.size))
        micro_reg_imgs = [None] * self.size

        for slide_obj in self.slide_dict.values():

            nr_obj = non_rigid_registrar.non_rigid_obj_dict[slide_obj.name]

            # Will be combining original and new dxdy as pyvips Images
            if not isinstance(nr_obj.bk_dxdy, pyvips.Image):
                vips_new_bk_dxdy = warp_tools.numpy2vips(np.dstack(nr_obj.bk_dxdy)).cast("float")
                vips_new_fwd_dxdy = warp_tools.numpy2vips(np.dstack(nr_obj.fwd_dxdy)).cast("float")
            else:
                vips_new_bk_dxdy = nr_obj.bk_dxdy
                vips_new_fwd_dxdy = nr_obj.fwd_dxdy

            if not isinstance(slide_obj.bk_dxdy[0], pyvips.Image):
                vips_current_bk_dxdy = warp_tools.numpy2vips(np.dstack(slide_obj.bk_dxdy)).cast("float")
                vips_current_fwd_dxdy = warp_tools.numpy2vips(np.dstack(slide_obj.fwd_dxdy)).cast("float")
            else:
                vips_current_bk_dxdy = slide_obj.bk_dxdy
                vips_current_fwd_dxdy = slide_obj.fwd_dxdy

            if np.any(non_rigid_registrar.shape != full_out_shape_rc):
                # Micro-registration performed on sub-region. Need to put in full image
                vips_new_bk_dxdy = self.pad_displacement(vips_new_bk_dxdy, full_out_shape_rc, mask_bbox_xywh)
                vips_new_fwd_dxdy = self.pad_displacement(vips_new_fwd_dxdy, full_out_shape_rc, mask_bbox_xywh)

            # Scale original dxdy to match scaled shape of new dxdy
            slide_sxy = (np.array(out_shape)/np.array([vips_current_bk_dxdy.height, vips_current_bk_dxdy.width]))[::-1]
            if not np.all(slide_sxy == 1):
                scaled_bk_dx = float(slide_sxy[0])*vips_current_bk_dxdy[0]
                scaled_bk_dy = float(slide_sxy[1])*vips_current_bk_dxdy[1]
                vips_current_bk_dxdy = scaled_bk_dx.bandjoin(scaled_bk_dy)
                vips_current_bk_dxdy = warp_tools.resize_img(vips_current_bk_dxdy, out_shape)

                scaled_fwd_dx = float(slide_sxy[0])*vips_current_fwd_dxdy[0]
                scaled_fwd_dy = float(slide_sxy[1])*vips_current_fwd_dxdy[1]
                vips_current_fwd_dxdy = scaled_fwd_dx.bandjoin(scaled_fwd_dy)
                vips_current_fwd_dxdy = warp_tools.resize_img(vips_current_fwd_dxdy, out_shape)

            vips_updated_bk_dxdy = vips_current_bk_dxdy + vips_new_bk_dxdy
            vips_updated_fwd_dxdy = vips_current_fwd_dxdy + vips_new_fwd_dxdy

            if not write_dxdy:
                # Will save numpy dxdy as Slide attributes
                np_updated_bk_dxdy = warp_tools.vips2numpy(vips_updated_bk_dxdy)
                np_updated_fwd_dxdy = warp_tools.vips2numpy(vips_updated_fwd_dxdy)

                slide_obj.bk_dxdy = np.array([np_updated_bk_dxdy[..., 0], np_updated_bk_dxdy[..., 1]])
                slide_obj.fwd_dxdy = np.array([np_updated_fwd_dxdy[..., 0], np_updated_fwd_dxdy[..., 1]])
            else:
                pathlib.Path(self.displacements_dir).mkdir(exist_ok=True, parents=True)
                slide_obj.stored_dxdy = True

                bk_dxdy_f, fwd_dxdy_f = slide_obj.get_displacement_f()
                slide_obj._bk_dxdy_f = bk_dxdy_f
                slide_obj._fwd_dxdy_f = fwd_dxdy_f

                # Save space by only writing the necessary areas. Most displacements may be 0
                cropped_bk_dxdy = vips_updated_bk_dxdy.extract_area(*mask_bbox_xywh)
                cropped_fwd_dxdy = vips_updated_fwd_dxdy.extract_area(*mask_bbox_xywh)

                cropped_bk_dxdy.cast("float").tiffsave(slide_obj._bk_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                cropped_fwd_dxdy.cast("float").tiffsave(slide_obj._fwd_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)

            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.processed_img
            else:
                img_to_warp = slide_obj.image

            micro_reg_img = slide_obj.warp_img(img_to_warp, non_rigid=True, crop=self.crop)


            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            micro_fout = os.path.join(self.micro_reg_dir, f"{img_save_id}_{slide_obj.name}.png")
            micro_thumb = self.create_thumbnail(micro_reg_img)
            warp_tools.save_img(micro_fout, micro_thumb)

            processed_micro_reg_img = slide_obj.warp_img(slide_obj.processed_img)
            micro_reg_imgs[slide_obj.stack_idx] = processed_micro_reg_img


        pickle.dump(self, open(self.reg_f, 'wb'))

        micro_overlap = self.draw_overlap_img(micro_reg_imgs)
        self.micro_reg_overlap_img = micro_overlap
        overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_micro_reg.png")
        warp_tools.save_img(overlap_img_fout, micro_overlap, thumbnail_size=self.thumbnail_size)

        print("\n==== Measuring error\n")
        error_df = self.measure_error()
        data_f_out = os.path.join(self.data_dir, self.name + "_summary.csv")
        error_df.to_csv(data_f_out, index=False)

        return non_rigid_registrar, error_df



