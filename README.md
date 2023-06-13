# OpenMPCudaCorrector

## Introduction

The project aims to comprehend and assess a specific code that exhibits poor performance and is written using OpenMP or CUDA. There are two stages for this project. The goal of the first stage is to detect and classify the performance bug class of the given code, and the second stage is the fixing of the given code. For performance-related problems, Azad et al., explained several performance bug classes, subclasses, and solutions to those bugs (Azad et al., 2023). For the classification part of this project, their definition of performance bugs is used. In order to accomplish the stated two objectives of the project, any AI model or tool can be utilized. However, since ChatGPT is recommended in the project proposal, it is decided to be used for both of the tasks. In addition, ChatGPT needs to be fine-tuned for having a model that can classify the performance bug of a given code and propose a solution to it.

## Preliminaries

ChatGPT is the most popular chatbot today. It has been released by the company called OpenAI on November 30, 2022. Chatbots have surely been present for many years, but a chatbot that can handle countless topics as well as ChatGPT has never been seen before (OpenAI, 2022).

This accomplishment is due to the improvements on large language models in recent years. Large language models are pre-trained transformer models that predict the probability of each token after a given text. There are currently many different large language models like LLAMA, GPT-3.5, GPT-4, BERT. In fact, GPT-4 can even work with multimodal input. It is able to understand image data as well as text data. However, we will work with the free version of ChatGPT which was fine-tuned on GPT-3.5 (Facebook, 2022; Devlin et al., 2018).

Construction of a chatbot such as ChatGPT is done by fine-tuning large language models into a model that responds to a given text. Billions of texts have been scraped from the internet by OpenAI, forming the training dataset. The performance has also been enhanced using a new method called RLHF. RLHF is an abbreviation of reinforcement learning human feedback, and as it can be understood from its name, it required human involvement. After the first training, the model returned different outputs for each input, then the best output has been chosen by human annotators. This hugely increased the performance and helped the model write human-level responses (Koubaa, 2023).

Even though it is a very powerful system as stated above, without fine-tuning the GPT model, ChatGPT could not always detect a performance problem in a given code. For example, in the Figure 1 below, an output of ChatGPT is represented for a given code.

Correct execution for performance fixing of ChatGPT
<img width="680" alt="ChatGPToutput01" src="https://github.com/KeremAydin98/OpenMPCudaCorrector/assets/77073029/bc3ee890-8be5-4477-a542-95d95aa792a9">
Figure 1: Correct execution for performance fixing of ChatGPT

From this figure, we can see that ChatGPT detects a performance problem in the code and proposes a correct solution that fixes it. However, when the code becomes a little bit more complicated, ChatGPT fails to detect a performance problem. An example of this is given in Figure 2.

Incorrect execution for performance fixing of ChatGPT
<img width="714" alt="ChatGPToutput02" src="https://github.com/KeremAydin98/OpenMPCudaCorrector/assets/77073029/e34cb6d4-20ec-4a8b-88d3-2d1dc5596ec6">
Figure 2: Incorrect execution for performance fixing of ChatGPT

OpenAI offers an API to use or fine-tune ChatGPT base models. Offered models are not as powerful as ChatGPT but still competent. In this project, we have focused on the following two tasks:

1. Detecting a class of problems in parallel programming, namely:
- False Sharing
- Inefficient Algorithm or Data Structure
- Inefficient Concurrency Control and Synchronization
- Inefficient Memory Management
- Missed Function Inlining
- Missing Parallelism
- Parallelization Overhead
- Poor Cache Utilization
- Unnecessary Branch
- Loop Unrolling

These classes are mentioned by Azad et al (Azad et al., 2023).

Creating the fixed version of a code which suffers from one of the problems mentioned above. As an example, let's consider the problem of false sharing. False sharing occurs when multiple threads modify different variables that share the same cache line in memory. There are two ways to solve this performance bug: either the memory layout can be changed using padding, or thread-aware data access can be formed.

Below is an example code that exhibits false sharing and its corresponding solution:

### False Sharing Bug
```
#include <omp.h>
#include <cstdio>
#include <iostream>

const int N = 1000000;
const int tNum = 4;

int main() {

    double start = omp_get_wtime();

    int data[N];
    int i;

    #pragma omp parallel num_threads(tNum)
    {
        int id = omp_get_thread_num();

        #pragma omp for
        for (i = id; i < N; i += tNum) {
            data[i] += 1;
        }

    }

    double time = omp_get_wtime() - start;

    printf("Elapsed time %f", time);

    return 0;
}
```

### False Sharing Solution
```
#include <omp.h>
#include <iostream>

const int N = 1000000;
const int tNum = 4;

int main() {

    double start = omp_get_wtime();

    // Pad each element to its own cache line
    int data[N][64];
    int i;

    #pragma omp parallel num_threads(tNum)
    {
        int id = omp_get_thread_num();
        #pragma omp for
        for (i = id; i < N; i += tNum) {
            data[i][0] += 1;
        }

    }

    double time = omp_get_wtime() - start;

    printf("Elapsed time %f", time);

    return 0;
}

```
