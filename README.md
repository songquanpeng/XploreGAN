# Exploring Unlabeled Faces for Novel Attribute Discovery

***This is an ongoing unofficial implementation.***

**Current progress:**
+ [x] Generate the pseudo labels.
+ [x] Verify the cluster results.
+ [x] Modify the dataloader.
+ [x] Implement the attribute summary instance normalization (ASIN).
+ [x] Modify the generator.
+ [x] Modify the discriminator.
+ [x] Implement the latent loss.
+ [ ] Generate fixed images per sample step.
+ [ ] Implement the test function.
+ [ ] Qualitative evaluation on CelebA.
+ [ ] Facial attribute translation on CelebA.
+ [ ] Image Translation between multiple datasets.

## Dependencies
```shell script
pip install -r requirements.txt

# Install faiss (optional)
# CPU version only
conda install faiss-cpu -c pytorch

# GPU version
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10

```

## Datasets
### CelebA
https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip

## Training & Testing
```sh
# Generate pseudo labels
python generate_labels.py --dataset_path D:\Research\Data\celeba\images \
                          --batch_size 32

# Inspect pseudo labels
python inspect_labels.py

# Train with the CelebA dataset
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 100 \
               --cluster_npz_path data/celeba/generated/clusters.npz \
               --batch_size 32

# Test with the CelebA dataset
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --cluster_npz_path data/celeba/generated/clusters.npz

# Use the pre-trainged network
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
               --model_save_dir='stargan_celeba_128/models' \
               --result_dir='stargan_celeba_128/results'
```


## Reference
1. The code is based on [StarGAN](https://github.com/yunjey/StarGAN).
2. https://arxiv.org/abs/1912.03085
3. https://github.com/wielandbrendel/bag-of-local-features-models