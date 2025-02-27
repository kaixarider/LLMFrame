#include "nccl_utils.h"
#include<stdio.h>


namespace util{
    void error_check(ncclResult_t result)
{
    if (result != ncclResult_t::ncclSuccess) {
        printf("[ERROR] NCCL error: %s\n", ncclGetErrorString(result));
        exit(-1);
    }
}

    void get_Comm_id(ncclUniqueId &nccl_id)
    {
        error_check(ncclGetUniqueId(&nccl_id));
    }
    NcclComm Nccl_init(
        int world_size,int rank,
        ncclUniqueId &nccl_id,cudaStream_t stream=0,
        bool real_init=true
    )
    {
        NcclComm new_comm;
        new_comm.rank=rank;
        new_comm.size=world_size;
        new_comm.stream=stream;
        if (world_size == 1 || !real_init) {
            new_comm.comm = nullptr;
            return new_comm;
        }
        error_check(ncclCommInitRank(&new_comm.comm, new_comm.size, nccl_id, new_comm.rank));
        return new_comm;
    }
    void Nccl_destroy(NcclComm &comm)
    {
        error_check(ncclCommDestroy(comm.comm));
    }
    void Nccl_reduce(
        void *sendbuff,
        void *recvbuff,
        int count,
        ncclDataType_t type,
        ncclRedOp_t op,
        NcclComm comm,
        cudaStream_t stream=0
    )
    {
        if(comm.comm==nullptr)
        {
            return;
        }
        error_check(ncclAllReduce(sendbuff,recvbuff,count,type,op,comm.comm,stream));
    }
    void Nccl_send(
        void* buff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t send_to,
        NcclComm comm,
        cudaStream_t stream = 0
    )
    {
        if(comm.comm==nullptr)
        {
            printf("Error! Nccl is not started.");
            return;
        }
        else if(send_to==comm.rank)
            {
                printf("Error!Can't send to self");
                return;
            }
        error_check(ncclSend(buff, count, datatype, send_to, comm.comm, stream));
    }

    void Nccl_recv(
        void* buff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t recv_from,
        NcclComm comm,
        cudaStream_t stream = 0
    ){
        if(comm.comm==nullptr)
        {
            printf("Error! Nccl is not started.");
            return;
        }
        else if(recv_from==comm.rank)
            {
                printf("Error!Can't send to self");
                return;
            }
        error_check(ncclRecv(buff, count, datatype, recv_from, comm.comm, stream));
    }
    void Nccl_send_and_recv(
        void* sendbuff,
        void* recvbuff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t send_rank,
        int64_t recv_rank,
        NcclComm comm,
        cudaStream_t stream = 0
    ){
        if(comm.comm==nullptr)
        {
            printf("Error! Nccl is not started.");
            return;
        }
        else if(recv_rank==send_rank)
        {
            printf("Error!Send rank and recv rank are the same!");
            return;
        }
        if(comm.rank==recv_rank)
            error_check(ncclRecv(recvbuff, count, datatype,recv_rank, comm.comm, stream));
        if(comm.rank==send_rank)
            error_check(ncclSend(sendbuff, count, datatype,recv_rank, comm.comm, stream));
    }

    void Nccl_broadcast(
        void* buff,
        int64_t count,
        ncclDataType_t datatype,
        int64_t root,
        NcclComm comm,
        cudaStream_t stream = 0
    ){
        if (comm.comm==nullptr)
        {
            printf("Error!Nccl is not established!");
            return;
        }
        error_check(ncclBcast(buff, count, datatype, root, comm.comm, stream));
    }
}