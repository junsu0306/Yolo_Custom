export CUDA_VISIBLE_DEVICES=3

python demo.py --demo ../images/CenterPoseTrack/shoe_test_copy.mp4 --arch dla_34 --load_model ../exp/object_pose/objectron_shoe_dla_34_2025-08-06-04-56/shoe_last.pth --tracking_task --debug 4