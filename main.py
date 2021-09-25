
if __name__ == '__main__':
  #load neccssary data
  #this part must be scaled to check if the model/ds have just been initialized(if so, don't load and get new data)
  model_vase6 = load_model('gdrive/MyDrive/ALIGNet/ALIGNet_model_vase6.pt').to(DEVICE)
  model_plane6 = load_model('gdrive/MyDrive/ALIGNet/ALIGNet_model_vase6.pt').to(DEVICE)
  tr_vase, val_vase, test_vase = load_ds('gdrive/MyDrive/ALIGNet/')
  tr_plane, val_plane, test_plane = load_ds('gdrive/MyDrive/ALIGNet/', ds_index=1)
  val_tar_vase, val_src_vase, val_tar_vase_aug, val_src_vase_aug = get_val_dl(tr_vase, val_vase, test_vase)
  val_tar_plane, val_src_plane, val_tar_plane_aug, val_src_plane_aug = get_val_dl(tr_plane, val_plane, test_plane)
  overfit_check_plane6 = overfit_checker(model_plane6,val_tar_plane, val_src_plane)
  train(model_plane6, 100, tr_plane, val_plane, test_plane,  'model_plane6',train_mode=0, graph_loss=True)
  validate(model_plane6, val_src_plane, val_tar_plane, GRID_SIZE, visualize = True, get_loss = True, checker_board=True)
