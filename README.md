Overview
--------

Caffe con Troll (CcT) is a clone of the uber popular Caffe framework
for Deep Learning. CcT is intended to be compatible with Caffe. We're
academics, which means that CcT is built for a research purpose: *to
explore the relative efficiency of GPUs and CPUs for Deep Learning*.

**Why Study CPU versus GPU?** Well, there is an ongoing debate about
this with lots of passion on both sides! GPU's are wildly [popular
some](http://www.wired.com/2015/02/hot-yet-little-known-trend-thatll-supercharge-ai/)
with many companies installing purpose-built infrastructure for deep
learning, but other
[companies](http://wired.com/2014/07/microsoft-adam/) have opted to
use CPUs. For users outside the web companies, the situation is
different: some cloud providers don't have GPUs or their GPUs are not
as rapidly updated as their CPUs. In the lab, GPUs can be expensive to
obtain.  In contrast, academic labs like ours have CPUs lying around
for other purposes, so we were curious about how much throughput we
could get from CPUs for Deep Learning. CcT is our first attempt. Our
initial results suggest that our CPU code is almost an order of
magnitude more efficient than Caffe's CPU code. In particular, two 8
core Haswells deliver roughly 80% of the throughput of a highest end
GPU (NVidia's Titan). At current consumer prices, which implies that,
chip-to-chip, the CPU solution costs almost 20% less than a GPU for
the same throughput. These numbers are *incredibly* rough but fun to
think about and troll our friends with!  Our GPU code is also slightly
faster than Caffe, see our [benchmark
page](http://deepdive.stanford.edu/cdw/benchmarking.html). This
speedup is mostly for non-fundamental reasons.

**New Technques** In the initial version, CcT's algorithms are
identical to Caffe from a statistical point of view. However, CcT uses
new lowering techniques to speed up convolutions and other layers
inspired by join processing in relational databases. As everyone
agrees, understanding the relational
[join](http://arxiv.org/abs/1310.3314) is the key to understanding the
universe. *Yes, that's just more trolling.* In the near future, we
plan to extend CcT in three directions but welcome feedback:

* Play with our
  [Hogwild!](http://i.stanford.edu/hazy/papers/hogwild-nips.pdf) ideas
  that are used in some of these frameworks.

* Explore the trade-off space described in our [DimmWitted
paper](http://arxiv.org/abs/1403.7550) in the context of Deep
Learning. In particular, we plan to use this framework to study the
trade off between statistical efficiency (*roughly, the number of
steps an algorithm take to converge*) and hardware efficiency
(*roughly, the efficiency of each of those steps*).

* Scale CcT to more than one machine. There are a host of challenges
  to cope with networking issues, delays, and maybe even faults.

* Integrate CcT with [DeepDive](http://deepdive.stanford.edu) to
  hopefully make it easier to build models and use them in
  applications.

If you have feedback let us know!

A Friendly VM
-------------

Probably the easiest way to try CcT is via a
[VM](http://deepdive.stanford.edu/cct/vm_page.html) that comes
preinstalled have an preinstalled with either MNIST (small and quick)
or ImageNet (big and fun). These VMs run on Amazon, Google Compute,
and Azure.

Installation from Source
------------------------

We cloned Caffe, so we follow nearly identical [install
instructions](http://caffe.berkeleyvision.org/installation.html).
Start with their instructions! One difference is that we only support
OpenBLAS (we don't support Atlas). We use some new features of
OpenBLAS, and so you'll want to use the included submodule in our git
repo.

* Step 1. Install the packages listed at the Caffe link.

* Step 2. Clone our repository including the OpenBLAS submodule.

> git clone --recursive https://github.com/HazyResearch/deepnet.git

* Step 3. Copy config.sample to .config and edit .config to contain your paths.

* Step 4. make && make test

It's good on a laptop, on a server, or for a snack. It is unclear
whether CcT can [smell the blood](http://en.wikipedia.org/wiki/Trollhunter) of christian men.

Contact
-------

Send flames to Chris and praise to those who did the actual work:
Firas Abuzaid, Shubham Gupta, and Ce Zhang.