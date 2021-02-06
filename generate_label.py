from pretrained import bagnet

model = bagnet.bagnet17(pretrained=True).cuda()
model.eval()

# First get all images' extracted features


# Then use k-means to cluster those features.
