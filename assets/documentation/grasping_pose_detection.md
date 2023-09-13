# Grasping Pose Detection (GPD)

```shell
conda activate contact_graspnet_env
python contact_graspnet/contact_graspnet_server.py
```

## View Grasping Poses

## API Reference for GPD Server


### Local Configuration

```http
  GET /run
```
###### Request

| Parameter | Type     | Description              |
|:----------| :------- |:-------------------------|
| `path`    | `string` | **Required**. Image data |

###### Response

| Parameter | Type   | Description                   |
|:----------|:-------|:------------------------------|
| `file`    | `FILE` | **Required**. predictions.npz |


### LAN Configuration

```http
  GET /run
```

###### Request

| Parameter | Type   | Description                 |
|:----------|:-------|:----------------------------|
| `file`    | `FILE` | **Required**. data.npy file |

###### Response

| Parameter | Type   | Description                   |
|:----------|:-------|:------------------------------|
| `file`    | `FILE` | **Required**. predictions.npz |







