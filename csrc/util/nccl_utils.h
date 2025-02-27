#pragma once

#include<nccl.h>
#include<mpi.h>
#include<stdlib.h>
#include<stdio.h>

namespace util{
    struct NcclComm{
        ncclComm_t comm;
        int rank;
        int size;
        cudaStream_t stream;
    NcclComm(){
        comm=nullptr;
        rank=0;
        size=1;
        stream=0;
    }
    };
    void get_Comm_id(ncclUniqueId &nccl_id);

    NcclComm Nccl_init(
        int world_size,int rank,
        ncclUniqueId &nccl_id,cudaStream_t stream=0,
        bool real_init=true
    );
    void Nccl_destroy(
        NcclComm & comm
    );
    void Nccl_reduce(
        void * sendbuff,
        void * recvbuff,
        int count,
        ncclDataType_t type,
        ncclRedOp_t op,
        NcclComm comm,
        cudaStream_t stream =0
    );
    
    void Nccl_send(
        void* buff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t send_to,
        NcclComm comm,
        cudaStream_t stream = 0
    );
    
    void Nccl_recv(
        void* buff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t recv_from,
        NcclComm comm,
        cudaStream_t stream = 0
    );
    
    void Nccl_send_and_recv(
        void* sendbuff,
        void* recvbuff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t send_rank,
        int64_t recv_rank,
        NcclComm comm,
        cudaStream_t stream = 0
    );
    
    void Nccl_broadcast(
        void* buff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t root,
        NcclComm comm,
        cudaStream_t stream = 0
    );
    
    template<typename T>
    ncclDataType_t Nccl_gettype()
    {
        ncclDataType_t nccl_data_type;
        if (std::is_same<T, float>::value) {
            nccl_data_type = ncclFloat32;
        }
        else if (std::is_same<T, half>::value) {
            nccl_data_type = ncclHalf;
        }
        else if (std::is_same<T, int>::value) {
            nccl_data_type = ncclInt;
        }
        else if (std::is_same<T, char>::value) {
            nccl_data_type = ncclChar;
        }
        else if (std::is_same<T, bool>::value) {
            nccl_data_type = ncclInt8;
        }
        else {
            printf("[ERROR] NCCL only support float, half, int, char, and bool. \n");
            exit(-1);
        }
        return nccl_data_type;
    }
    
}