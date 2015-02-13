Overview
--------

Caffe-DimmWitted (CDW) is a clone of the uber popular Caffe framework
for Deep Learning. CDW is intended to be fully compatible with
Caffe. We're academics, so we really built CDW for research purposes:
*to explore the relative efficiency of GPUs and CPUs for Deep
Learning*. Since not all cloud providers have GPUs, GPUS can be
expensive to obtain in large numbers, and we have large numbers of
CPUs lying around the lab, we were curious about how much throughput
we could get from CPUs for Deep Learning. This code is our first
attempt. Although we do our best, it's alpha software.

Our initial results suggest that our CPU code is almost an order of
magnitude more efficient than Caffe's version. In particular, two 8
core Haswells deliver about as much throughput as a Titan GPU. At
current consumer prices, that means that, chip-to-chip, the CPU
solution costs almost 40% less than a GPU for the same throughput. Our
GPU code is also slightly faster than Caffe, see our [benchmark
page](http://deepdive.stanford.edu/cdw/benchmarking.html). Right now,
CDW's default algorithm is identical to Caffe from a statistical point
of view. However, CDW uses some new lowering techniques inspired by
join processing in relational databases. As everyone agrees,
understanding the relational [join](http://arxiv.org/abs/1310.3314) is
the key to understanding the universe. *Yes, that's just trolling.*

In the near future, we plan to extend CDW in three directions:

* More fully explore the trade offs from our [DimmWitted
paper](http://arxiv.org/abs/1403.7550) in the context of Deep
Learning. In particular, we plan to use this framework to study the
trade off between statistical efficiency (*how many steps does the
algorithm take to converge*) versus hardware efficiency (*how
efficient are each of those steps*). We'll play with our [Hogwild!
ideas](http://i.stanford.edu/hazy/papers/hogwild-nips.pdf).

* Scale it out to more than one machine. There are a host of
  challenges to cope with networking issues.

* Integrate CDW with [DeepDive](http://deepdive.stanford.edu) in the
  hopes of making it easier to build these models and use them in
  application.

If you have feedback let us know!

A Friendly VM
-------------

Probably the easiest way to try it out is using a
[VM](http://deepdive.stanford.edu/cdw/vm_page.html) that comes
preinstalled have an preinstalled with either MNIST (small and quick)
or ImageNet (big and fun). These VMs run on Amazon, Google Compute, and
Azure.

Installation from Source
------------------------

We cloned Caffe, so we follow nearly identical [install
instructions](http://caffe.berkeleyvision.org/installation.html) so
start there! One difference is that we only support OpenBLAS. We muck
around with OpenBLAS internals, and so you'll want to use the included
submodule.

* Step 1. Install the packages listed at the Caffe link.

* Step 2. Clone our repository including the OpenBLAS submodule.

> git clone --recursive https://github.com/HazyResearch/deepnet.git

* Step 3. Copy config.sample to .config and edit .config to contain your paths.

* Step 4. make && make test

It's good on a laptop, on a server, or for a snack.

Contact
-------

Send flames to Chris and praise to those who did the actual work:
Firas Abuzaid, Shubham Gupta, and Ce Zhang.