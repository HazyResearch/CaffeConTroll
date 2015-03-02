<img src="cct_logo.png" width=200 align=left></img>

Caffe con Troll (CcT) is a clone of the uber popular Caffe framework
for Deep Learning. CcT is intended to be compatible with Caffe. We're
academics, which means that CcT is built for a research purpose: *to
explore the relative efficiency of GPUs and CPUs for Deep
Learning*. However, in our system you can use both at once!

**Why Study CPU versus GPU?** Well, there is an ongoing debate about
this with lots of passion on both sides! GPU's are wildly [popular
with
some](http://www.wired.com/2015/02/hot-yet-little-known-trend-thatll-supercharge-ai/)
companies that are rumored to be installing purpose-built
infrastructures for deep learning; other
[companies](http://wired.com/2014/07/microsoft-adam/) have opted to
use CPUs and claimed they are cheaper and more efficient. For users
outside the web companies, the situation is different: some cloud
providers don't have GPUs or their GPUs are not as rapidly updated as
their CPUs. In the lab, GPUs can be expensive to obtain.  In contrast,
academic labs like ours have CPUs lying around for other purposes, so
we were curious about how much throughput we could get from CPUs for
Deep Learning. CcT is our first attempt. Our initial results suggest
that our CPU code is almost an order of magnitude more efficient than
Caffe's CPU code. In particular, on Amazon, two 8-core haswells
deliver the same throughput as the GPU. And three 8-core Haswells
deliver roughly the throughput of a highest end GPU (NVidia's
Titan). Since GPU instances have GPUs and CPUs, we can go even faster
by combining the two! 

**New Techniques** In the initial version of CcT, CcT's algorithms are
identical to Caffe from a statistical point of view. However, CcT uses
new lowering techniques to speed up convolutions and other layers
inspired by join processing in relational databases. As everyone
agrees, understanding the relational
[join](http://arxiv.org/abs/1310.3314) is the key to understanding the
universe. *Yes, that's just more trolling.* In the near future, we
plan to extend CcT in a few directions:

* Play with our
  [Hogwild!](http://i.stanford.edu/hazy/papers/hogwild-nips.pdf) ideas
  that are used in some of deep learning frameworks.

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

Of course, if you have feedback or challenge problems, let us know!

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
Start with their instructions! *NB: the .travis.yml contains a working
build script for Ubuntu 12.04, if you are confused about
dependencies.*


* Step 1. Install the packages listed at the Caffe link.

* Step 2. Clone our repository 

> git clone git@github.com:HazyResearch/CaffeConTroll.git

* Step 3. Copy config.sample to .config and edit .config to contain your paths.

* Step 4. Build the executable.

> make

* Step 5. (Optional) If you want tests, you need to install Google's
testing infrastructure, glog and gtest, as with Caffe. Then, make the
test file.

> make test && ./test


It's good on a laptop, on a server, or for a snack. We strive for [Morgan's
Maxim](http://en.wikipedia.org/wiki/Poe%27s_law), _"Any sufficiently advanced troll is indistinguishable from a genuine kook."_

Known Issues
------------
* It is unclear whether CcT can [smell the
blood](http://en.wikipedia.org/wiki/Trollhunter) of christian
men.

* OpenBLAS seems to be the fastest, free library. However, OpenBLAS
  has some known issues on Mac. It will compile, but it will unpredictably
  [crash](https://github.com/xianyi/OpenBLAS/issues/218). On recent OS
  X, you can use the built in BLAS and LAPACK (see Caffe install
  instruction.)

* If you use Vanilla BLAS, ATLAS, or the installed Apple's BLAS
  libraries CcT may be slower.
  
Contact
-------

Send flames to Chris and praise to those who did the actual work:
Firas Abuzaid, Shubham Gupta, and Ce Zhang.