# mako.dev

## interview prep:
- study RL to improve models (might take some time (a project or blog post on this will do.. just code which you can clone or even run a docker image to kick models off training on their own))
- do some more architectural planning and be absolutely delusional while its happening -> for auto project
- touch up on actual kernel code (go through simons blog post again and relearn all those kernels (should take a day if i really put my mind to it (2-3 hrs per kernel up to kernel 6 (vectorized i think) ) ) ) 
- skills: rl, kernel coding

potential competitors:
- mojo
- luminal

Positioning in the market:
- Mojo exists -> we can automate mojo kernel generation as well, but cuda/rocm/triton/cutlass is still unsolved problem
- B2B or B2C?
- we may end up pivoting to a company that does not generate deep learning kernels. There are lots of different ways that you can apply GPUs to problems. 
- We currently have no product market fit. I don't know if it's best to work internally from a company and develop that way or develop externally and try to sell. We need people who want to buy this. We need the distribution as our moat.
-  I'm currently unsure how validity works, like how people test internally within labs, just like getting those thoughts from others and their processes versus ours and how easy it would be to do with a small team. And like maybe what internal tools you would need or to even build out for that. 
- Focuses may not be in the realm of how can we get this to run faster and inference. They might already have inference and it would be like really hard to generate something that would be faster than those kernels and that would be like overhead for them. But instead it might be transitioning over to like a DSL or transitioning to AMD hardware, in which case you have different problems there. You now have to worry about AMD. That is also something that we can solve if we get AMD GPUs. 
- If you decide to build internally inside of a company and it fails, then you still have the job. But if you succeed, then the company gets billions of dollars. As opposed to if you build externally, you probably have lower chances and you have to change your approach a little bit because of the connections that you have internally versus externally. If it does really well, you make billions of dollars. And if it doesn't, then you kind of flop and you have to figure out something else or pivot. 
- best to dedicate a week to this project after maybe we could do this over time throughout this cohort, maybe after my book is done when I get back to Edmonton, and we can try to dedicate a day to just accumulating all of the research in this area and just really getting familiar with it again, and then really objectifying our approach the next day, developing sort of an execution plan on what we want to do given all the information that we got the first day, and then we spend the next five days literally executing on that and simply moving forward with getting like a Docker container with GPUs to run and just like recursively improve itself or even to just like whatever we want to get sort of like a very quick MVP going. 
- Start with smaller fish. 
- build in public and open source. 
- the question I would have is that, like, is it trivial to just go and generate more kernels, take all the different types of them and just go make them faster? Like, you have convolutions and layer norms and matmuls and all these, and these are very common, and you can apply more optimizations to those, but like, for instance, quantization, like, it might be very simple, and at the end of the day, just like a very simple threat operation that is, like, already coarsened, and you, like, can't really improve that much. I dont know what the scope of these optimizations are, and we might already be limited to, like, a select few, but I don't know this yet. 

Compile and benchmark on modal (modal.com) gpus so I dont require a shit ton of time on gpus for no reason at all. Ideally with everything setup beforehand and also bare metal.

Open source or not?
- Yes -> get customers through here than optimize their kernels

Look up to:
- https://github.com/ScalingIntelligence/scalingintelligence.github.io
- https://x.com/mako_dev_ai/status/1937873917646897479
- Manus.im 
- openmanus


