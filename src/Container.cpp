//
//  Container.cpp
//  xCCT
//
//  Created by Igor B on 5/15/15.
//  Copyright (c) 2015 Igor B. All rights reserved.
//

#include "Container.h"
#include "Layer.h"
#
#define myLog(a) //std::cout <<__FILE__<<": "<< __PRETTY_FUNCTION__<<": "<<__LINE__ <<": "<<a<<"\n"



Container::Container(int _myRank,int _mB,int _iD,int _iR,int _iC, cnn::LayerParameter * _p_layer_param):
	myRank(_myRank),mB(_mB),iD(_iD),iR(_iR),iC(_iC),p_layer_param(_p_layer_param)
	{
    	if (p_layer_param->type()==cnn::LayerParameter_LayerType_CONVOLUTION)
        {
            //Compute all dimensions:
            k = p_layer_param->convolution_param().kernel_size(),
            p = p_layer_param->convolution_param().pad(),
            s = p_layer_param->convolution_param().stride();
            oR = compute_conv_next_layer_dimension(iR, k, p, s),
            oC = compute_conv_next_layer_dimension(iC, k, p, s),
            oD = p_layer_param->convolution_param().num_output();
            //Allocate Cubes:
            //p_model = new LogicalCube<float, Layout_CRDB>(k, k, iD, oD);
            //p_bias = new LogicalCube<float, Layout_CRDB>(1, 1, oD, 1);
            p_X = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
            p_Y = new LogicalCube<float, Layout_CRDB>(oR, oC, oD, mB);
            p_dX = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
            p_dY = new LogicalCube<float, Layout_CRDB>(oR, oC, oD, mB);
            //Allocate layers:
            p_input_layer = new Layer<float, Layout_CRDB>(p_X,p_dX);
            p_output_layer = new Layer<float, Layout_CRDB>(p_Y,p_dY);
			//Configure solver: TODO: Should solver be passed to Container constructor?
            p_solver_param = new cnn::SolverParameter() ;
            p_solver_param->set_base_lr(0.01);
            p_solver_param->set_momentum(0.0);
            p_solver_param->set_lr_policy("step");
            p_solver_param->set_stepsize(10000);
            //Allocate driver:
            CPUDriver * p_scheduler_local_cpudriver = new CPUDriver();
            //Set to all CPU
            p_layer_param->set_gpu_0_batch_proportion(0); //TODO: enable GPU as well
            //Allocate Bridge
			p_bridge = new ConvolutionBridge<float, Layout_CRDB, float, Layout_CRDB, CPUDriver>(p_input_layer,p_output_layer,p_layer_param,p_solver_param,p_scheduler_local_cpudriver);
            
            //Set Grad+bias updaters: //TODO: this is common to different types of bridges
            float model_base_learning_rate=1.0;
            float model_base_regularization=1.0;
            if (p_layer_param->blobs_lr_size() != 0) {
                model_base_learning_rate = p_layer_param->blobs_lr(0);
            }
            if (p_layer_param->weight_decay_size() != 0) {
                model_base_regularization = p_layer_param->weight_decay(0);
            }
            p_grad_updater = new SGDGradientUpdater<float, CPUDriver>(p_bridge->p_model_cube_shadow->n_elements, p_bridge->p_model_cube_shadow->get_p_data(),p_solver_param, model_base_learning_rate, model_base_regularization, p_scheduler_local_cpudriver);
            p_grad_updater_bias = new SGDGradientUpdater<float, CPUDriver>(p_bridge->p_bias_cube->n_elements, p_bridge->p_bias_cube->get_p_data(),p_solver_param, model_base_learning_rate, model_base_regularization, p_scheduler_local_cpudriver);
            
            
        }
        else if (p_layer_param->type()==cnn::LayerParameter_LayerType_INNER_PRODUCT)
        {

            oR = oC = 1;
            oD = p_layer_param->inner_product_param().num_output();
            
            //p_model = new LogicalCube<float, Layout_CRDB>(k, k, iD, oD);
            //p_bias = new LogicalCube<float, Layout_CRDB>(1, 1, oD, 1);
            p_X = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
            p_Y = new LogicalCube<float, Layout_CRDB>(oR, oC, oD, mB);
            p_dX = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
            p_dY = new LogicalCube<float, Layout_CRDB>(oR, oC, oD, mB);
            p_input_layer = new Layer<float, Layout_CRDB>(p_X,p_dX);
            p_output_layer = new Layer<float, Layout_CRDB>(p_Y,p_dY);
            //Configure solver: TODO: Should solver be passed to Container constructor?
            p_solver_param = new cnn::SolverParameter() ;
            p_solver_param->set_base_lr(0.01);
            p_solver_param->set_momentum(0.0);
            p_solver_param->set_lr_policy("step");
            p_solver_param->set_stepsize(10000);
            //Allocate driver:
            CPUDriver * p_scheduler_local_cpudriver = new CPUDriver();
            //Set to all CPU
            p_layer_param->set_gpu_0_batch_proportion(0); //TODO: enable GPU as well
            //Allocate Bridge
            p_bridge = new ConvolutionBridge<float, Layout_CRDB, float, Layout_CRDB, CPUDriver>(p_input_layer,p_output_layer,p_layer_param,p_solver_param,p_scheduler_local_cpudriver);
            
            //Set Grad+bias updaters: //TODO: this is common to different types of bridges
            float model_base_learning_rate=1.0;
            float model_base_regularization=1.0;
            if (p_layer_param->blobs_lr_size() != 0) {
                model_base_learning_rate = p_layer_param->blobs_lr(0);
            }
            if (p_layer_param->weight_decay_size() != 0) {
                model_base_regularization = p_layer_param->weight_decay(0);
            }
            p_grad_updater = new SGDGradientUpdater<float, CPUDriver>(p_bridge->p_model_cube_shadow->n_elements, p_bridge->p_model_cube_shadow->get_p_data(),p_solver_param, model_base_learning_rate, model_base_regularization, p_scheduler_local_cpudriver);
            p_grad_updater_bias = new SGDGradientUpdater<float, CPUDriver>(p_bridge->p_bias_cube->n_elements, p_bridge->p_bias_cube->get_p_data(),p_solver_param, model_base_learning_rate, model_base_regularization, p_scheduler_local_cpudriver);
        }

}



         


void Container::run(){
    int batchID=0;
    for (int j =0; j<1000; j++) {
        
        batchID=feed(p_bridge->p_input_layer->p_data_cube,ANY_BATCH,FEATURES);
        p_bridge->forward();
        evict(p_bridge->p_output_layer->p_data_cube, batchID, FEATURES);
        
        batchID=feed(p_bridge->p_output_layer->p_gradient_cube,ANY_BATCH,GRADS); 
        p_bridge->backward();
        p_grad_updater->update(p_bridge->get_model_grad_cube()->get_p_data()); //TODO: this should be in backward()
        p_grad_updater_bias->update(p_bridge->get_bias_grad_cube()->get_p_data());
        
        evict(p_bridge->p_input_layer->p_gradient_cube, batchID, GRADS);
    }	
}





void Container::receiveCube(LogicalCube<float, Layout_CRDB>* p_input, int src_rank) {
    myLog(p_input->get_p_data());
    MPI_Recv(p_input->get_p_data(), (int) p_input->n_elements, MPI_FLOAT, src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void Container::sendCube(LogicalCube<float, Layout_CRDB>* p_output, int dst_rank) {
    MPI_Send(p_output->get_p_data(), (int) p_output->n_elements, MPI_FLOAT, dst_rank, 0,MPI_COMM_WORLD);
}

Container::~Container() {
    free (p_model);
    free (p_bias);
    free ( p_X);
    free (p_Y);
    free (p_dX);
    free (p_dY);
    free (p_input_layer);
    free (p_output_layer);
    free (p_layer_param);
    free (p_conv_param);
    free (p_solver_param);
    free (p_bridge);

}

//#define MSG_SIZE 6
//enum requestType {EVICT,FEED,SYNC};
//enum dataClass {LABEL,FEATURES,GRAD,MODEL,BIAS};
//enum other {ANY_BATCH=-1,SCHED=0,METADATA};

void printMsg(int msg[]) {
    
    for (int i=0; i<MSG_SIZE; i++)
    {
        std::cout<<msg[i]<<" ";
    }
    std::cout<<"\n";
}
int Container::feedModel() {
    int outMsg[MSG_SIZE]={myRank,FEED,0,MODEL,METADATA,requestID}; //1,1,0, 1,0
    printMsg(outMsg);
    requestID++;
    int inMsg[MSG_SIZE];
    MPI_Send(outMsg, MSG_SIZE, MPI_INT, SCHED, 1,MPI_COMM_WORLD);
    MPI_Recv(inMsg, MSG_SIZE, MPI_INT, SCHED, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int srcRank=inMsg[0];

    MPI_Recv(p_bridge->p_model_cube_shadow->get_p_data(), (int) p_bridge->p_model_cube_shadow->n_elements, MPI_FLOAT, srcRank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return 0;
}


int Container::feed(LogicalCube<float, Layout_CRDB>* p_input, int batch_id,int dataType) {
    int outMsg[MSG_SIZE]={myRank,FEED,batch_id,dataType,METADATA,requestID}; //1,1,-1,1,1,1
    //std::cout<<myRank<<" sending: "; printMsg(outMsg);
    requestID++;
    int inMsg[MSG_SIZE];
    MPI_Send(outMsg, MSG_SIZE, MPI_INT, SCHED, 1,MPI_COMM_WORLD);
    MPI_Recv(inMsg, MSG_SIZE, MPI_INT, SCHED, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //std::cout<<myRank<<" received: "; printMsg(inMsg);
    int srcRank=inMsg[0];
    int batchID=inMsg[2];
    MPI_Recv(p_input->get_p_data(), (int) p_input->n_elements, MPI_FLOAT, srcRank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return batchID;
}

int Container::evict(LogicalCube<float, Layout_CRDB>* p_input, int batch_id,int dataType) {
    int outMsg[MSG_SIZE]={myRank,EVICT,batch_id,dataType,METADATA,requestID};
    //std::cout<<myRank<<" sending: "; printMsg(outMsg);
    requestID++;
    int inMsg[MSG_SIZE];
    MPI_Send(outMsg, MSG_SIZE, MPI_INT, SCHED, 1,MPI_COMM_WORLD);
    MPI_Recv(inMsg, MSG_SIZE, MPI_INT, SCHED, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //std::cout<<myRank<<" received: "; printMsg(inMsg);
    int dstRank=inMsg[0];
    //std::cout<<myRank<<" evicting "<< p_input->n_elements<<" elements\n";
    MPI_Send(p_input->get_p_data(), (int) p_input->n_elements, MPI_FLOAT, dstRank, 2, MPI_COMM_WORLD);
    return 0;
}



