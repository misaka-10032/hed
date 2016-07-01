#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForward(const int n,
                                               const int dim,
                                               const Dtype* in,  // this is BEFORE sigmoid
                                               const Dtype* tgt,
                                               const Dtype* loss_pw,
                                               const Dtype* loss_nw,
                                               Dtype* loss) {
  CUDA_KERNEL_LOOP(i, n) {
    const int bi = i / dim;  // batch index
    loss[i] = in[i] * (tgt[i] - (in[i] >= 0)) -
        log(1 + exp(in[i] - 2 * in[i] * (in[i] >= 0)));
    loss[i] *= tgt[i] > 0.5 ? loss_pw[bi] : loss_nw[bi];
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossBackward(const int n,
                                                const int dim,
                                                const Dtype* out,  // this is AFTER sigmoid
                                                const Dtype* tgt,
                                                const Dtype scale,
                                                const Dtype* loss_pw,
                                                const Dtype* loss_nw,
                                                Dtype* din) {
  CUDA_KERNEL_LOOP(i, n) {
    const int bi = i / dim;  // batch index
    Dtype diff = out[i] - tgt[i];
    diff *= scale;
    diff *= tgt[i] > 0.5 ? loss_pw[bi] : loss_nw[bi];
    din[i] = diff;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //LOG(INFO) << "Before pos_cnt";
  // count pos/neg labels for each example
  // we assume a small batch size (num) here
  const int bs = bottom[0]->num();
  const int dim = bottom[0]->count() / bs;
  for (int i = 0; i < bs; i++) {  // iterate over batch idx
    // it's two-class label, so the sum is the num of pos
    Dtype pos_cnt = Dtype(0);
    caffe_gpu_asum(dim, bottom[1]->gpu_data()+i*dim, &pos_cnt);
    cudaDeviceSynchronize();  // TODO: need it?
    Dtype nw = pos_cnt / dim;
    loss_nw_.mutable_cpu_data()[i] = nw;
    loss_pw_.mutable_cpu_data()[i] = Dtype(1) - nw;
  }
  //LOG(INFO) << "After pos_cnt";

  //LOG(INFO) << "Before fwd";
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  Dtype* loss_pw = loss_pw_.mutable_gpu_data();
  Dtype* loss_nw = loss_nw_.mutable_gpu_data();
  Dtype* loss_data = loss_data_.mutable_gpu_data();
  SigmoidCrossEntropyLossForward
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, dim, input_data, target, loss_pw, loss_nw, loss_data);
  CUDA_POST_KERNEL_CHECK;
  //LOG(INFO) << "After fwd";

  Dtype loss = Dtype(0);
  caffe_gpu_asum(count, loss_data, &loss);
  cudaDeviceSynchronize();  // TODO: need it?
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    //LOG(INFO) << "Before bwd";
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int dim = count / num;
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    const Dtype scale = top[0]->cpu_diff()[0] / num;
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* loss_pw = loss_pw_.gpu_data();
    const Dtype* loss_nw = loss_nw_.gpu_data();
    SigmoidCrossEntropyLossBackward
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (count, dim, sigmoid_output_data, target,
         scale, loss_pw, loss_nw, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    //LOG(INFO) << "After bwd";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
