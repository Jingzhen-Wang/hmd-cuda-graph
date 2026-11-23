/*----------------------------------------------------------------------
Program hmd_graph.cu -- Phase 3 of the hmd-cuda-graph final project:
adds CUDA Graph capture/replay on top of Phase 2's validated full
MPI + OpenMP (one thread per GPU) + CUDA structure. This is the
project's central optimization: instead of re-issuing the same ~7
host<->device operations (3 copies in, 2 memsets, 1 kernel launch, 2
copies out) via separate CPU API calls on every single one of the
thousands of compute_accel() calls a run makes, each thread captures
that fixed sequence ONCE into a cudaGraphExec_t and replays it with a
single cudaGraphLaunch() on every subsequent call -- amortizing the
per-call CPU-side launch overhead that Phase 1/2's naive per-step
issuing pays every time.

Everything except compute_accel() is unchanged from Phase 2 (hmd_gpu.cu),
including gpu_compute_accel_kernel() itself -- Phase 3 only changes HOW
that same kernel (and its surrounding copies) gets launched, never WHAT
it computes, so Phase 2's validated numerics carry over unchanged.

Modifications relative to Phase 2, all marked "***** MODIFIED (GRAPH) *****":

1. Per-thread device buffers are now sized to their COMPILE-TIME MAXIMA
   (NEMAX/NMAX/NCLMAX) and copied/zeroed at that fixed size on EVERY
   call, never at the current, run-time-varying n/nb/lcxyz2. This is
   required for graph replay to be safe: n and nb (resident/boundary
   atom counts) change slightly over the run as atoms cross MPI-rank
   boundaries in atom_move()/atom_copy(), but a captured graph node's
   byte count is frozen at capture time -- replaying a node captured for
   a smaller n against a later, larger n would silently copy too little
   data. Copying the full fixed extent instead means every graph node's
   size is a compile-time constant, so the SAME graph is correct and
   reusable for the rest of the run no matter how n/nb drift. (This is
   safe, not just convenient: head[]/lscl[]'s linked lists only ever
   point to indices below the current n+nb, so the kernel never reads
   the extra, possibly-stale slots copied above that range. lcxyz2 only
   depends on al[]/RCUT, which this fixed-volume NVE run never changes,
   so it is already constant across the whole run.)
2. Each thread creates its own cudaStream_t once (so later kernel
   launches/copies are issued on that stream rather than the default
   stream) and, on its first call, records the fixed operation sequence
   with cudaStreamBeginCapture(..., cudaStreamCaptureModeThreadLocal) /
   cudaStreamEndCapture(), then cudaGraphInstantiate()s it into a
   cudaGraphExec_t kept in a static per-thread array (paralleling
   dev_r[]/dev_ra[]/etc.). ThreadLocal capture mode specifically allows
   NUM_DEVICE threads to each be mid-capture on their own stream/device
   at the same time without one thread's unrelated CUDA calls being
   mistaken for another's capture.
3. compute_accel() itself now just does cudaGraphLaunch(graphExec[std],
   stream[std]) followed by cudaStreamSynchronize(stream[std]) on every
   call after the first -- no cudaMemcpy/cudaMemset/kernel-launch calls
   appear in the per-step path at all. Because every captured node reads
   from / writes to the SAME fixed host and device addresses every call
   (the host-side r[]/head[]/lscl[] arrays are not reallocated -- single_step()
   / atom_move() / atom_copy() update them in place), replaying the graph
   automatically picks up each step's fresh data; no node-parameter update
   (cudaGraphExecKernelNodeSetParams) or recapture is ever needed.
----------------------------------------------------------------------*/
#include "hmd.h"

#define MAX_THREADS 64  /* ***** MODIFIED (GPU) ***** cap for per-thread
                            device-buffer arrays; nthrd (default 4, from
                            vthrd[]) is always far below this. */

static int num_device = 1;  /* ***** MODIFIED (GPU) ***** # of GPUs
                                visible on this node; set once in main() */

/* ***** MODIFIED (GPU) ***** pair-force kernel: one thread block per
   interior linked-list cell owned by the calling OpenMP thread
   (blockIdx.{x,y,z}+1+mofst{0,1,2} = mc[0..2]), unchanged from Phase 1
   otherwise. */
__global__ void gpu_compute_accel_kernel(
    const double *r, double *ra, const int *head, const int *lscl,
    double *pe_cell, int lcyz2, int lc2d2, const int *n_ptr,
    double rrCut, double Duc, double Uc,
    int mofst0, int mofst1, int mofst2)
{
  /* ***** MODIFIED (GRAPH) ***** n is read through a pointer, not passed
     as a captured scalar argument: n (the resident-atom count) can
     change between graph capture and a later replay as atoms cross
     MPI-rank boundaries, but a captured kernel node's scalar arguments
     are frozen at capture time. Reading *n_ptr instead lets the host
     update the pointed-to value before every replay (same pattern as
     r[]/head[]/lscl[]: fixed address, fresh contents). */
  int n = *n_ptr;
  int mc0 = blockIdx.x + 1 + mofst0;
  int mc1 = blockIdx.y + 1 + mofst1;
  int mc2 = blockIdx.z + 1 + mofst2;
  int c = mc0*lcyz2 + mc1*lc2d2 + mc2;

  double lpe = 0.0;

  if (head[c] != EMPTY) {
    for (int d0 = -1; d0 <= 1; d0++)
    for (int d1 = -1; d1 <= 1; d1++)
    for (int d2 = -1; d2 <= 1; d2++) {
      int c1 = (mc0+d0)*lcyz2 + (mc1+d1)*lc2d2 + (mc2+d2);
      if (head[c1] == EMPTY) continue;

      int ii = head[c];
      while (ii != EMPTY) {
        int jj = head[c1];
        while (jj != EMPTY) {
          if (jj != ii) {
            double dr0 = r[3*ii+0]-r[3*jj+0];
            double dr1 = r[3*ii+1]-r[3*jj+1];
            double dr2 = r[3*ii+2]-r[3*jj+2];
            double rr = dr0*dr0+dr1*dr1+dr2*dr2;
            if (rr < rrCut) {
              double ri2 = 1.0/rr, ri6 = ri2*ri2*ri2, r1v = sqrt(rr);
              double fcVal = 48.0*ri2*ri6*(ri6-0.5) + Duc/r1v;
              /* Safe without atomics: ii's home cell is c, owned solely
                 by this block, so ra[3*ii+*] is never touched by any
                 other block. */
              ra[3*ii+0] += fcVal*dr0;
              ra[3*ii+1] += fcVal*dr1;
              ra[3*ii+2] += fcVal*dr2;
              if (ii < jj) {
                double vVal = 4.0*ri6*(ri6-1.0) - Uc - Duc*(r1v-RCUT);
                lpe += (jj < n) ? vVal : 0.5*vVal;
              }
            }
          }
          jj = lscl[jj];
        }
        ii = lscl[ii];
      }
    }
  }
  pe_cell[c] = lpe;
}

/*--------------------------------------------------------------------*/
int main(int argc, char **argv) {
/*--------------------------------------------------------------------*/
  double cpu1;

  MPI_Init(&argc,&argv); /* Initialize the MPI environment */
  MPI_Comm_rank(MPI_COMM_WORLD, &sid);  /* My processor ID */
  /* Vector index of this processor */
  vid[0] = sid/(vproc[1]*vproc[2]);
  vid[1] = (sid/vproc[2])%vproc[1];
  vid[2] = sid%vproc[2];

  /* ***** MODIFIED (GPU) ***** restore OpenMP (one thread per GPU, as in
     hmd_cpu.c), and find out how many GPUs this node actually has */
  omp_set_num_threads(nthrd);
  cudaGetDeviceCount(&num_device);
  if (num_device < 1) num_device = 1;
  if (sid == 0) printf("nthrd = %d, num_device = %d\n", nthrd, num_device);

  init_params();
  set_topology();
  init_conf();
  atom_copy();
  compute_accel(); /* Computes initial accelerations */

  cpu1 = MPI_Wtime();
  for (stepCount=1; stepCount<=StepLimit; stepCount++) {
    single_step();
    if (stepCount%StepAvg == 0) eval_props();
  }
  cpu = MPI_Wtime() - cpu1;
  if (sid == 0) printf("CPU & COMT = %le %le\n",cpu,comt);

  MPI_Finalize(); /* Clean up the MPI environment */
  return 0;
}

/*--------------------------------------------------------------------*/
void init_params() {
/*----------------------------------------------------------------------
Initializes parameters.  Unchanged from hmd_cpu.c: thbk[]/vthrd[] are
still computed so lc[]/rc[] match the CPU reference's geometry exactly,
even though this phase's compute_accel() no longer partitions work by
thread block.
----------------------------------------------------------------------*/
  int a;
  double rr,ri2,ri6,r1;
  FILE *fp;

  /* Read control parameters */
  fp = fopen("pmd.in","r");
  fscanf(fp,"%d%d%d",&InitUcell[0],&InitUcell[1],&InitUcell[2]);
  fscanf(fp,"%le",&Density);
  fscanf(fp,"%le",&InitTemp);
  fscanf(fp,"%le",&DeltaT);
  fscanf(fp,"%d",&StepLimit);
  fscanf(fp,"%d",&StepAvg);
  fclose(fp);

  /* Compute basic parameters */
  DeltaTH = 0.5*DeltaT;
  for (a=0; a<3; a++) al[a] = InitUcell[a]/pow(Density/4.0,1.0/3.0);
  if (sid == 0) printf("al = %e %e %e\n",al[0],al[1],al[2]);

  /* Compute the # of cells for linked cell lists */
  for (a=0; a<3; a++) {
    lc[a] = al[a]/RCUT;          /* Cell size >= potential cutoff */
    thbk[a] = lc[a]/vthrd[a];    /* unused by the GPU kernel; kept so lc[]
                                     matches the CPU reference exactly */
    lc[a] = thbk[a]*vthrd[a];    /* Adjust # of cells to a multiple of vthrd */
    rc[a] = al[a]/lc[a];
  }
  if (sid == 0) {
    printf("lc = %d %d %d\n",lc[0],lc[1],lc[2]);
    printf("rc = %e %e %e\n",rc[0],rc[1],rc[2]);
  }

  /* Constants for potential truncation */
  rr = RCUT*RCUT; ri2 = 1.0/rr; ri6 = ri2*ri2*ri2; r1=sqrt(rr);
  Uc = 4.0*ri6*(ri6 - 1.0);
  Duc = -48.0*ri6*(ri6 - 0.5)/r1;
}

/*--------------------------------------------------------------------*/
void set_topology() {
/*----------------------------------------------------------------------
Defines a logical network topology.  Prepares a neighbor-node ID table,
nn, & a shift-vector table, sv, for internode message passing.  Also
prepares the node parity table, myparity.
----------------------------------------------------------------------*/
  /* Integer vectors to specify the six neighbor nodes */
  int iv[6][3] = {
    {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
  };
  int ku,a,k1[3];

  /* Set up neighbor tables, nn & sv */
  for (ku=0; ku<6; ku++) {
    /* Vector index of neighbor ku */
    for (a=0; a<3; a++)
      k1[a] = (vid[a]+iv[ku][a]+vproc[a])%vproc[a];
    /* Scalar neighbor ID, nn */
    nn[ku] = k1[0]*vproc[1]*vproc[2]+k1[1]*vproc[2]+k1[2];
    /* Shift vector, sv */
    for (a=0; a<3; a++) sv[ku][a] = al[a]*iv[ku][a];
  }

  /* Set up the node parity table, myparity */
  for (a=0; a<3; a++) {
    if (vproc[a] == 1)
      myparity[a] = 2;
    else if (vid[a]%2 == 0)
      myparity[a] = 0;
    else
      myparity[a] = 1;
  }
}

/*--------------------------------------------------------------------*/
void init_conf() {
/*----------------------------------------------------------------------
r are initialized to face-centered cubic (fcc) lattice positions.
rv are initialized with a random velocity corresponding to Temperature.
----------------------------------------------------------------------*/
  double c[3],gap[3],e[3],vSum[3],gvSum[3],vMag;
  int j,a,nX,nY,nZ;
  double seed;
  /* FCC atoms in the original unit cell */
  double origAtom[4][3] = {{0.0, 0.0, 0.0}, {0.0, 0.5, 0.5},
                           {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};

  /* Set up a face-centered cubic (fcc) lattice */
  for (a=0; a<3; a++) gap[a] = al[a]/InitUcell[a];
  n = 0;
  for (nZ=0; nZ<InitUcell[2]; nZ++) {
    c[2] = nZ*gap[2];
    for (nY=0; nY<InitUcell[1]; nY++) {
      c[1] = nY*gap[1];
      for (nX=0; nX<InitUcell[0]; nX++) {
        c[0] = nX*gap[0];
        for (j=0; j<4; j++) {
          for (a=0; a<3; a++)
            r[n][a] = c[a] + gap[a]*origAtom[j][a];
          ++n;
        }
      }
    }
  }
  /* Total # of atoms summed over processors */
  MPI_Allreduce(&n,&nglob,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (sid == 0) printf("nglob = %d\n",nglob);

  /* Generate random velocities */
  seed = 13597.0+sid;
  vMag = sqrt(3*InitTemp);
  for(a=0; a<3; a++) vSum[a] = 0.0;
  for(j=0; j<n; j++) {
    RandVec3(e,&seed);
    for (a=0; a<3; a++) {
      rv[j][a] = vMag*e[a];
      vSum[a] = vSum[a] + rv[j][a];
    }
  }
  MPI_Allreduce(vSum,gvSum,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  /* Make the total momentum zero */
  for (a=0; a<3; a++) gvSum[a] /= nglob;
  for (j=0; j<n; j++)
    for(a=0; a<3; a++) rv[j][a] -= gvSum[a];
}

/*--------------------------------------------------------------------*/
void single_step() {
/*----------------------------------------------------------------------
r & rv are propagated by DeltaT using the velocity-Verlet scheme.
----------------------------------------------------------------------*/
  int i,a;

  half_kick(); /* First half kick to obtain v(t+Dt/2) */
  for (i=0; i<n; i++) /* Update atomic coordinates to r(t+Dt) */
    for (a=0; a<3; a++) r[i][a] = r[i][a] + DeltaT*rv[i][a];
  atom_move();
  atom_copy();
  compute_accel(); /* Computes new accelerations, a(t+Dt) */
  half_kick(); /* Second half kick to obtain v(t+Dt) */
}

/*--------------------------------------------------------------------*/
void half_kick() {
/*----------------------------------------------------------------------
Accelerates atomic velocities, rv, by half the time step.
----------------------------------------------------------------------*/
  int i,a;
  for (i=0; i<n; i++)
    for (a=0; a<3; a++) rv[i][a] = rv[i][a]+DeltaTH*ra[i][a];
}

/*--------------------------------------------------------------------*/
void atom_copy() {
/*----------------------------------------------------------------------
Exchanges boundary-atom coordinates among neighbor nodes:  Makes
boundary-atom list, LSB, then sends & receives boundary atoms.
----------------------------------------------------------------------*/
  int kd,kdd,i,ku,inode,nsd,nrc,a;
  int nbnew = 0; /* # of "received" boundary atoms */
  double com1;
  MPI_Request request;

/* Main loop over x, y & z directions starts--------------------------*/

  for (kd=0; kd<3; kd++) {

    /* Make a boundary-atom list, LSB---------------------------------*/

    /* Reset the # of to-be-copied atoms for lower&higher directions */
    for (kdd=0; kdd<2; kdd++) lsb[2*kd+kdd][0] = 0;

    /* Scan all the residents & copies to identify boundary atoms */
    for (i=0; i<n+nbnew; i++) {
      for (kdd=0; kdd<2; kdd++) {
        ku = 2*kd+kdd; /* Neighbor ID */
        /* Add an atom to the boundary-atom list, LSB, for neighbor ku
           according to bit-condition function, bbd */
        if (bbd(r[i],ku)) lsb[ku][++(lsb[ku][0])] = i;
      }
    }

    /* Message passing------------------------------------------------*/

    com1=MPI_Wtime(); /* To calculate the communication time */

    /* Loop over the lower & higher directions */
    for (kdd=0; kdd<2; kdd++) {

      inode = nn[ku=2*kd+kdd]; /* Neighbor node ID */

      /* Send & receive the # of boundary atoms-----------------------*/

      nsd = lsb[ku][0]; /* # of atoms to be sent */

      /* Asynchronous exchange; no even/odd parity ordering needed */
      MPI_Irecv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,10,MPI_COMM_WORLD,&request);
      MPI_Send(&nsd,1,MPI_INT,inode,10,MPI_COMM_WORLD);

      /* Independent computation while nrc is in flight:
         message buffering does not depend on the received data */
      for (i=1; i<=nsd; i++)
        for (a=0; a<3; a++) /* Shift the coordinate origin */
          dbuf[3*(i-1)+a] = r[lsb[ku][i]][a]-sv[ku][a];

      MPI_Wait(&request,&status);
      /* Now nrc is the # of atoms to be received */

      /* Send & receive information on boundary atoms-----------------*/

      MPI_Irecv(dbufr,3*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,20,MPI_COMM_WORLD,
                &request);
      MPI_Send(dbuf,3*nsd,MPI_DOUBLE,inode,20,MPI_COMM_WORLD);
      MPI_Wait(&request,&status);

      /* Message storing */
      for (i=0; i<nrc; i++)
        for (a=0; a<3; a++) r[n+nbnew+i][a] = dbufr[3*i+a];

      /* Increment the # of received boundary atoms */
      nbnew = nbnew+nrc;

      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);

    } /* Endfor lower & higher directions, kdd */

    comt += MPI_Wtime()-com1; /* Update communication time, COMT */

  } /* Endfor x, y & z directions, kd */

  /* Main loop over x, y & z directions ends--------------------------*/

  /* Update the # of received boundary atoms */
  nb = nbnew;
}

/*--------------------------------------------------------------------*/
void compute_accel() {
/*----------------------------------------------------------------------
Given atomic coordinates, r[0:n+nb-1][], for the extended (i.e.,
resident & copied) system, computes the acceleration, ra[0:n-1][], for
the residents.

***** MODIFIED (GRAPH) *****
The linked-cell-list build (head[]/lscl[]) stays on the host, unchanged
from hmd_cpu.c/hmd_gpu.c.  Per OpenMP thread, the fixed sequence of
device operations (copy in, zero, kernel, copy out) is captured ONCE
into a CUDA Graph and replayed via cudaGraphLaunch() on every call
thereafter -- see the file header for why this is both correct (every
node's byte count is a compile-time maximum, never the run-time-varying
n/nb) and sufficient (no per-step parameter update is needed, since
every node reads/writes the same fixed addresses the host code keeps
updating in place). Each thread's results land in its own host buffer
and are merged into the shared ra[]/lpe_td[] sequentially after the
parallel region closes, exactly as in Phase 2.
----------------------------------------------------------------------*/
  int i,a,lc2[3],lcyz2,lcxyz2,mc[3],c,std;
  double lpe;
  double lpe_td[MAX_THREADS];

  static double *dev_r[MAX_THREADS]    = {NULL};
  static double *dev_ra[MAX_THREADS]   = {NULL};
  static double *dev_pe[MAX_THREADS]   = {NULL};
  static int    *dev_head[MAX_THREADS] = {NULL};
  static int    *dev_lscl[MAX_THREADS] = {NULL};
  static double *ra_host_td[MAX_THREADS] = {NULL};
  static double *pe_host_td[MAX_THREADS] = {NULL};
  static int    *dev_n[MAX_THREADS]    = {NULL}; /* ***** MODIFIED (GRAPH) *****
                                                     device shadow of n, see kernel comment */
  static int     n_td[MAX_THREADS];               /* host staging value for dev_n[std] */
  static cudaStream_t    stream[MAX_THREADS];
  static cudaGraphExec_t graphExec[MAX_THREADS];

  /* Reset the forces (host) & per-thread energy accumulators */
  for (i=0; i<n; i++) for (a=0; a<3; a++) ra[i][a] = 0.0;
  for (std=0; std<nthrd; std++) lpe_td[std] = 0.0;

  /* Make a linked-cell list, lscl (host, unchanged from hmd_cpu.c)----*/

  for (a=0; a<3; a++) lc2[a] = lc[a]+2;
  lcyz2 = lc2[1]*lc2[2];
  lcxyz2 = lc2[0]*lcyz2;

  for (c=0; c<lcxyz2; c++) head[c] = EMPTY;

  for (i=0; i<n+nb; i++) {
    for (a=0; a<3; a++) mc[a] = (r[i][a]+rc[a])/rc[a];
    c = mc[0]*lcyz2+mc[1]*lc2[2]+mc[2];
    lscl[i] = head[c];
    head[c] = i;
  }

  /* ***** MODIFIED (GRAPH) ***** begin: one OpenMP thread per GPU, each
     driving its own cell block on its own device, via a captured CUDA
     Graph instead of per-step API calls -- mirrors Phase 2's parallel
     region (std/vtd[]/mofst[] computed the same way). */
  #pragma omp parallel private(std)
  {
    int vtd[3], mofst[3], a2;

    std = omp_get_thread_num();
    vtd[0] = std/(vthrd[1]*vthrd[2]);
    vtd[1] = (std/vthrd[2])%vthrd[1];
    vtd[2] = std%vthrd[2];
    for (a2=0; a2<3; a2++) mofst[a2] = vtd[a2]*thbk[a2];

    cudaSetDevice(std % num_device);

    /* ***** MODIFIED (GRAPH) ***** refresh the host staging value for
       dev_n[std] every call (capture and replay alike); the graph's
       memcpy node has a fixed source address (&n_td[std]) but picks up
       whatever is written here before cudaGraphLaunch. */
    n_td[std] = n;

    if (!dev_r[std]) {
      /* First call for this thread: allocate fixed-size device buffers
         (same as Phase 2) and host mirrors, then CAPTURE the fixed
         operation sequence into a graph instead of just running it. */
      cudaMalloc((void**)&dev_r[std],    sizeof(double)*NEMAX*3);
      cudaMalloc((void**)&dev_ra[std],   sizeof(double)*NMAX*3);
      cudaMalloc((void**)&dev_head[std], sizeof(int)*NCLMAX);
      cudaMalloc((void**)&dev_lscl[std], sizeof(int)*NEMAX);
      cudaMalloc((void**)&dev_pe[std],   sizeof(double)*NCLMAX);
      cudaMalloc((void**)&dev_n[std],    sizeof(int));
      ra_host_td[std] = (double*)malloc(sizeof(double)*NMAX*3);
      pe_host_td[std] = (double*)malloc(sizeof(double)*NCLMAX);

      cudaStreamCreate(&stream[std]);

      /* dim3 grid is fixed for the whole run: thbk[]/mofst[] never change
         after init_params(), so the captured launch configuration (and
         every node's addresses/sizes) stays valid on every later replay. */
      dim3 grid(thbk[0], thbk[1], thbk[2]);

      cudaStreamBeginCapture(stream[std], cudaStreamCaptureModeThreadLocal);
      /* ***** MODIFIED (GRAPH) ***** fixed, compile-time-maximum byte
         counts on every node -- see file header point 1. */
      cudaMemcpyAsync(dev_r[std],    r,    sizeof(double)*NEMAX*3, cudaMemcpyHostToDevice, stream[std]);
      cudaMemcpyAsync(dev_head[std], head, sizeof(int)*NCLMAX,     cudaMemcpyHostToDevice, stream[std]);
      cudaMemcpyAsync(dev_lscl[std], lscl, sizeof(int)*NEMAX,      cudaMemcpyHostToDevice, stream[std]);
      cudaMemcpyAsync(dev_n[std],   &n_td[std], sizeof(int),       cudaMemcpyHostToDevice, stream[std]);
      cudaMemsetAsync(dev_ra[std], 0, sizeof(double)*NMAX*3,  stream[std]);
      cudaMemsetAsync(dev_pe[std], 0, sizeof(double)*NCLMAX,  stream[std]);
      gpu_compute_accel_kernel<<<grid,1,0,stream[std]>>>(
          dev_r[std], dev_ra[std], dev_head[std], dev_lscl[std], dev_pe[std],
          lcyz2, lc2[2], dev_n[std], RCUT*RCUT, Duc, Uc,
          mofst[0], mofst[1], mofst[2]);
      cudaMemcpyAsync(ra_host_td[std], dev_ra[std], sizeof(double)*NMAX*3,  cudaMemcpyDeviceToHost, stream[std]);
      cudaMemcpyAsync(pe_host_td[std], dev_pe[std], sizeof(double)*NCLMAX,  cudaMemcpyDeviceToHost, stream[std]);

      cudaGraph_t graph;
      cudaStreamEndCapture(stream[std], &graph);
      cudaGraphInstantiate(&graphExec[std], graph, NULL, NULL, 0);
      cudaGraphDestroy(graph); /* graphExec[std] keeps its own copy */
    }

    /* Every call, first and later alike: one CPU-side call replaces the
       ~7 separate API calls Phase 2 issued here. */
    cudaGraphLaunch(graphExec[std], stream[std]);
    cudaStreamSynchronize(stream[std]);
  } /* End omp parallel */

  /* Sequential merge (see file header for why this isn't done above):
     each thread's host buffer is zero everywhere except the atoms/cells
     it owns, so a straight add reproduces hmd_cpu.c's shared-array
     result exactly, with no cross-thread races. */
  for (std=0; std<nthrd; std++) {
    for (i=0; i<n; i++) for (a=0; a<3; a++) ra[i][a] += ra_host_td[std][3*i+a];
    for (c=0; c<lcxyz2; c++) lpe_td[std] += pe_host_td[std][c];
  }

  lpe = 0.0;
  for (std=0; std<nthrd; std++) lpe += lpe_td[std];
  /* ***** MODIFIED (GRAPH) ***** end */

  /* Global potential energy */
  MPI_Allreduce(&lpe,&potEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

/*--------------------------------------------------------------------*/
void eval_props() {
/*----------------------------------------------------------------------
Evaluates physical properties: kinetic, potential & total energies.
----------------------------------------------------------------------*/
  double vv,lke;
  int i,a;

  /* Total kinetic energy */
  for (lke=0.0, i=0; i<n; i++) {
    for (vv=0.0, a=0; a<3; a++) vv += rv[i][a]*rv[i][a];
    lke += vv;
  }
  lke *= 0.5;
  MPI_Allreduce(&lke,&kinEnergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

  /* Energy paer atom */
  kinEnergy /= nglob;
  potEnergy /= nglob;
  totEnergy = kinEnergy + potEnergy;
  temperature = kinEnergy*2.0/3.0;

  /* Print the computed properties */
  if (sid == 0) printf("%9.6f %9.6f %9.6f %9.6f\n",
                stepCount*DeltaT,temperature,potEnergy,totEnergy);
}

/*--------------------------------------------------------------------*/
void atom_move() {
/*----------------------------------------------------------------------
Sends moved-out atoms to neighbor nodes and receives moved-in atoms
from neighbor nodes.  Called with n, r[0:n-1] & rv[0:n-1], atom_move
returns a new n' together with r[0:n'-1] & rv[0:n'-1].
----------------------------------------------------------------------*/

/* Local variables------------------------------------------------------

mvque[6][NBMAX]: mvque[ku][0] is the # of to-be-moved atoms to neighbor
  ku; MVQUE[ku][k>0] is the atom ID, used in r, of the k-th atom to be
  moved.
----------------------------------------------------------------------*/
  int mvque[6][NBMAX];
  int newim = 0; /* # of new immigrants */
  int ku,kd,i,kdd,kul,kuh,inode,ipt,a,nsd,nrc;
  double com1;
  MPI_Request request;

  /* Reset the # of to-be-moved atoms, MVQUE[][0] */
  for (ku=0; ku<6; ku++) mvque[ku][0] = 0;

  /* Main loop over x, y & z directions starts------------------------*/

  for (kd=0; kd<3; kd++) {

    /* Make a moved-atom list, mvque----------------------------------*/

    /* Scan all the residents & immigrants to list moved-out atoms */
    for (i=0; i<n+newim; i++) {
      kul = 2*kd  ; /* Neighbor ID */
      kuh = 2*kd+1;
      /* Register a to-be-copied atom in mvque[kul|kuh][] */
      if (r[i][0] > MOVED_OUT) { /* Don't scan moved-out atoms */
        /* Move to the lower direction */
        if (bmv(r[i],kul)) mvque[kul][++(mvque[kul][0])] = i;
        /* Move to the higher direction */
        else if (bmv(r[i],kuh)) mvque[kuh][++(mvque[kuh][0])] = i;
      }
    }

    /* Message passing with neighbor nodes----------------------------*/

    com1 = MPI_Wtime();

    /* Loop over the lower & higher directions------------------------*/

    for (kdd=0; kdd<2; kdd++) {

      inode = nn[ku=2*kd+kdd]; /* Neighbor node ID */

      /* Send atom-number information---------------------------------*/

      nsd = mvque[ku][0]; /* # of atoms to-be-sent */

      /* Asynchronous exchange; no even/odd parity ordering needed */
      MPI_Irecv(&nrc,1,MPI_INT,MPI_ANY_SOURCE,110,MPI_COMM_WORLD,&request);
      MPI_Send(&nsd,1,MPI_INT,inode,110,MPI_COMM_WORLD);

      /* Independent computation while nrc is in flight:
         message buffering does not depend on the received data */
      for (i=1; i<=nsd; i++)
        for (a=0; a<3; a++) {
          /* Shift the coordinate origin */
          dbuf[6*(i-1)  +a] = r [mvque[ku][i]][a]-sv[ku][a];
          dbuf[6*(i-1)+3+a] = rv[mvque[ku][i]][a];
        }

      MPI_Wait(&request,&status);
      /* Now nrc is the # of atoms to be received */

      /* Send & receive information on boundary atoms-----------------*/

      MPI_Irecv(dbufr,6*nrc,MPI_DOUBLE,MPI_ANY_SOURCE,120,MPI_COMM_WORLD,
                &request);
      MPI_Send(dbuf,6*nsd,MPI_DOUBLE,inode,120,MPI_COMM_WORLD);

      /* Independent computation while atom data is in flight:
         mark the moved-out atoms (moved here from the buffering loop) */
      for (i=1; i<=nsd; i++)
        r[mvque[ku][i]][0] = MOVED_OUT; /* Mark the moved-out atom */

      MPI_Wait(&request,&status);

      /* Message storing */
      for (i=0; i<nrc; i++)
        for (a=0; a<3; a++) {
          r [n+newim+i][a] = dbufr[6*i  +a];
          rv[n+newim+i][a] = dbufr[6*i+3+a];
        }

      /* Increment the # of new immigrants */
      newim = newim+nrc;

      /* Internode synchronization */
      MPI_Barrier(MPI_COMM_WORLD);

    } /* Endfor lower & higher directions, kdd */

    comt=comt+MPI_Wtime()-com1;

  } /* Endfor x, y & z directions, kd */

  /* Main loop over x, y & z directions ends--------------------------*/

  /* Compress resident arrays including new immigrants */

  ipt = 0;
  for (i=0; i<n+newim; i++) {
    if (r[i][0] > MOVED_OUT) {
      for (a=0; a<3; a++) {
        r [ipt][a] = r [i][a];
        rv[ipt][a] = rv[i][a];
      }
      ++ipt;
    }
  }

  /* Update the compressed # of resident atoms */
  n = ipt;
}

/*----------------------------------------------------------------------
Bit condition functions:

1. bbd(ri,ku) is .true. if coordinate ri[3] is in the boundary to
     neighbor ku.
2. bmv(ri,ku) is .true. if an atom with coordinate ri[3] has moved out
     to neighbor ku.
----------------------------------------------------------------------*/
int bbd(double* ri, int ku) {
  int kd,kdd;
  kd = ku/2; /* x(0)|y(1)|z(2) direction */
  kdd = ku%2; /* Lower(0)|higher(1) direction */
  if (kdd == 0)
    return ri[kd] < RCUT;
  else
    return al[kd]-RCUT < ri[kd];
}
int bmv(double* ri, int ku) {
  int kd,kdd;
  kd = ku/2; /* x(0)|y(1)|z(2) direction */
  kdd = ku%2; /* Lower(0)|higher(1) direction */
  if (kdd == 0)
    return ri[kd] < 0.0;
  else
    return al[kd] < ri[kd];
}
