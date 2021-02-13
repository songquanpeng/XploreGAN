# Exploring Unlabeled Faces for Novel Attribute Discovery

***This is an ongoing unofficial implementation.***
Current progress:
+ [x] Generate the pseudo labels.
+ [x] Verify the cluster results.
+ [ ] Implement the proposed model.

## Dependencies
`pip install -r requirements.txt`

## Datasets
### CelebA
https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip

## Training & Testing (Need update)
```sh
# Generate labels
python generate_labels.py --dataset_path D:\Research\Data\celeba\images 
                         --batch_size 32

# Inspect generated labels
python inspect_labels.py

# Train StarGAN using the CelebA dataset
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

# Test StarGAN using the CelebA dataset
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

# Use pre-trainged network
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
               --model_save_dir='stargan_celeba_128/models' \
               --result_dir='stargan_celeba_128/results'
```


## Reference
1. The code is based on [StarGAN](https://github.com/yunjey/StarGAN).
2. https://arxiv.org/abs/1912.03085
3. https://github.com/wielandbrendel/bag-of-local-features-models