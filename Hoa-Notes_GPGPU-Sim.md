# GPGPU-Sim Notes
The following notes focus on the architecture simulation of GPGPU-Sim, namely the execution pipeline of a single SIMT Core. Higher level memory (greater than L1), clocks, and scheduling will not be considered. In addition, optimization details like branch decisions, stalls, and barriers are not the focus, rather we are trying to understand the core functionality of GPU hardware being simulated in GPGPU-Sim.

## Simulated Architecture

### High Level Design
GPGPU Sim splits into 2 main programs, cuda-sim and gpgpu-sim, providing a functional and performance (execution) simulations respectively. Nvidia graphics cards are composed of streaming multiprocessors (SM) that contain caches, registers, and execution hardware. In GPGPU-Sim, a single instruction multiple thread (SIMT) core mimics the Nvidia SM, similarly composed of caches and registers. Each SIMT core executes one thread block, and is scheduled using round robin.

SIMT core cycle (software implementation): fetch > decode > issue > read_operand > execute > writeback

This core cycle represents the functions called in GPGPU-Sim during a simulation and do not represent the actual stages of hardware pipeline. In other words, each function does not take one clock cycle. Having said that, the execute stage can often take multiple cycles due to complex operations or memory access.

---

#### Front End (fetch + decode + issue)
The front end represents the part of the architecture pipeline before the ALU, revolving around instruction fetching.

##### Fetch
There is an I-Buffer and an I-Cache. The I-Buffer contains instructions fetched from the I-Cache. When a warp cannot execute any instructions in the I-Buffer (empty, invalid), a fetch is called to read instructions from the I-Cache. Note that once a warp is finished executing (no pending operations), it no longer fetches instructions.

##### Decode
Once an instruction is fetched from the I-Cache, it is decoded and stored into the I-Buffer, ready to be issued.

##### Issue
Round-robin scheduler is used to pick a *valid* instruction from the I-Buffer to issue. *Valid* means that the current warp is not waiting at a barrier, the valid bit is activated for the instruction, it passes the scoreboard check, and the next stage in the pipeline is not stalled.

##### SIMT Stack
A per-warp SIMT Stack handles branch divergence. It is connected to the fetch unit to update the PC target address. This stack is updated after each instruction issue.

##### Scoreboard
The scoreboard keeps track of WAW (write after write) and RAW (read after write) hazards.

#### Backend (Collector + ALU + MEM)
The backend represent the execution part of the architecture pipeline, revolving around execution and register handling.

##### Register Access and Operand Collector
A collector unit buffers the source operands (that map to a register) of an instruction after it has been decoded. When the ready bits are set in the collector unit, the instruction can then be sent to the execution unit. An arbiter can withold register access (due to write-backs or conflicts), preventing the instructions from being sent. The arbiter and collector unit work together to manage register access so that only when all operands for an instruction are ready, can it execute. This forces one register bank access per cycle. Note that in GPGPU-Sim, the backend pipelines (SP, SFU, and MEM) each have their own collector unit that share a general collector unit. 


##### ALU
SIMT core contains two functional units, SFU and SP, for transcendentals (sine, cosine, log, etc.) and non-transcendentals respectively. The SFU takes more cycles than the SP. Both units share a writeback stage, but due the operand collector connection, a shared write-back should never occur.

##### Memory
There are 4 L1 memories supported in GPGPU-Sim: Shared (R/W), Constant (R), Texture (R), and Data (R/W). 

---

### Kernel Execution Example
Once a kernel is launched, a grid is used to represent the work for that kernel. This grid is composed of thread blocks that each execute on an SIMT Core. That is one block for one core. At the SIMT Core level, the blocks are divided into warps, which are a group of threads that execute in parallel. Each thread contains instructions. Once a warp is scheduled for a core, the I-Buffer (which is initially empty or invalid) will fill up with fresh decoded instructions loaded from the I-Cache. (This should consume 2 clock cycles?). In the following clock cycle, the instructions will flow into the collector unit, waiting to be issued. Since this is the first instruction, the registers should all be ready, so the arbiter will have set the valid bits, and the instruction and it's operands flow into the ALU. Whether this is a memory or non-memory access instruction, the proper unit will take the neccessary clock cycles to perform the operation and the result will flow into the writeback stage. The writeback stage involves the arbiter once again, which will give it priority over the upcoming instructions to update the register file. Assuming no stalls, barriers, or branches, this will complete the execution of one instruction. 

---

## Software
The two main programs, cuda-sim and gpgpu-sim, provide functional and performance (execution) simulations respectively. The functional simulation interprets binaries (PTX instructions) compiled by NVCC, and executes them functionally. The performance simulation executes instructions based on the state of the functional simulation. 

The SIMT core cluster class, *simt_core_cluster*, contains an array of SIMT Core classes, *shader_core_ctx*. The cluster's responsibility is to inject packets (block?) containing instructions to each SIMT core's I-Cache. 

The class representing a SIMT core is *shader_core_ctx*. This class provides the implementation for the simulated architecture pipeline (fetch(), decode(), issue(), etc.). It contains a member varialbe *m_thread* which is an array of *ptx_thread_info*, representing all the active threads of the simulated SIMT core.

The I-Cache stores a array of *shd_warp_t* objects, which contains a set of *ibuffer_entry* objects. 
The I-Buffer, represented by *m_ibuffer*, stores *ibuffer_entry* objects.

The fetch() function of a *shader_core_ctx* class grabs *shd_warp_t* objects stored in the I-Cache. Once an entry is decoded, it stores a pointer to a *warp_inst_t* object.

The *warp_inst_t* class represents an instance of an instruction being executed by a single warp, composed of the type of operation and operands used. Once the performance simulator executes an instruction, the functional simulator updates the thread state. 

## src/gpgpu-sim

## src/cuda-sim

## Glossary

**Streaming Multiprocessor (SM)**: A processor capable of highly multi-threaded execution, complete with registers, caches (shared memory, L1, texture, etc.), and execution units (tensor core, integer, double, etc.).

**Single Instruction Multiple Thread (SIMT) Core**: GPGPU sim equivalent of an Nvidia SM.

**Kernel**: A C++ function.

**Grid**: A group of blocks, launched per kernel.

**Thread Block**: A group of warps that share SM/SIMT resources. (1 SM per Block?)

**Warp**: A unit of execution, composed of 32 threads that act in lockstep (parallel). When a warp stalls in a SM/SIMT, another warp is loaded. 