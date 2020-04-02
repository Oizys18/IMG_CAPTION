



```
    np.savetxt(
        './datasets/train_datasets.csv', train_dataset, fmt='%s', delimiter='|'
    )
    np.save('./datasets/train_datasets.npy', train_dataset)
```

```
    data = np.loadtxt(dataset_path, delimiter='|', dtype=np.str)
    data = np.load(dataset_path)
```

