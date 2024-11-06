# llama.cpp for TOPSCC

 - [Background](#background)
 - [News](#news)
 - [OS](#os)
 - [Hardware](#hardware)
 - [Model Supports](#model-supports)
 - [DataType Supports](#datatype-supports)
 - [Docker](#docker)
 - [Linux](#linux)
 - [TODO](#todo)


## Background

**GCU**

**TOPSCC**

**Llama.cpp + TOPSCC**

## News

## OS

| OS      | Status  | Verified                                       |
|:-------:|:-------:|:----------------------------------------------:|
|    |  |                    |


## Hardware

### GCU

**Verified devices**
| GCU                    | Status  |
|:-----------------------------:|:-------:|
|                  |  |

## Model Supports

| Model Name                  | FP16  | Q8_0 | Q4_0 |
|:----------------------------|:-----:|:----:|:----:|
|               |      |     |     |




## DataType Supports

| DataType               | Status  |
|:----------------------:|:-------:|
|                    |  |

## Docker

## Linux

### I. Setup Environment

### II. Build llama.cpp

```sh
cmake -B build -DGGML_TOPSCC=on -DCMAKE_BUILD_TYPE=release
cmake --build build --config release
```

### III. Run the inference

## TODO
