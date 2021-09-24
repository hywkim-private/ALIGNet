#train the data
#data_index 0 = vase, 1= plane
#iter is the number of iterations
#train_mode = 0: only trains on original images
#train_mode = 1: only trains on augmented images
#train_mode = 2: trains both on original and augmented images
def train(model, iter, tr, val, test, model_name, train_mode = 0, overfit_checker=None, graph_loss=False):
  for i in range(iter):
    if train_mode == 0:
      tr_tar_ds, tr_src_ds, val_tar_ds, val_src_ds, test_tar_ds, test_src_ds = aug_datasets(tr, val, test)

      tr_tar_dl = get_dataloader(tr_tar_ds, augment=True, shuffle=True)
      tr_src_dl = get_dataloader(tr_src_ds, shuffle=True)
      tr_tar_ds  = pre_augment(tr_tar_dl, batch_size=BATCH_SIZE, shuffle=True)
      tr_tar_dl = DataLoader(tr_tar_ds, BATCH_SIZE, shuffle=True)


      tr_tar = tr_tar_dl
      tr_src = tr_src_dl
    if train_mode == 1 or train_mode == 2:
      tr_tar_ds_aug, tr_src_ds_aug, val_tar_ds_aug, val_src_ds_aug, test_tar_ds_aug, test_src_ds_aug = aug_datasets(tr, val, test, augment=True)
      tr_tar_dl_aug = get_dataloader(tr_tar_ds_aug, augment=True, shuffle=True)
      tr_src_dl_aug = get_dataloader(tr_src_ds_aug, shuffle=True)
      #we will pre-process(augment) the data in order to prevent overloading gpu
      tr_tar_ds_aug  = pre_augment(tr_tar_dl_aug, batch_size=BATCH_SIZE, shuffle=True)
      tr_tar_dl_aug = DataLoader(tr_tar_ds_aug, BATCH_SIZE, shuffle=True)
      if train_mode == 1:
        tr_tar = [tr_tar_dl, tr_tar_dl_aug]
        tr_src = [tr_src_dl,tr_src_dl_aug]
      elif train_mode == 2:
        tr_tar = tr_tar_dl_aug
        tr_src = tr_src_dl_aug

    run_model(model, tr_src, tr_tar, GRID_SIZE, overfit_checker = overfit_checker, graph_loss=graph_loss)
    torch.save(model_plane6.state_dict(), 'gdrive/MyDrive/ALIGNet/ALIGNet_'+model_name+'.pt')