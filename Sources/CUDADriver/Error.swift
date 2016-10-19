//
//  Error.swift
//  CUDA
//
//  Created by Richard Wei on 10/16/16.
//
//

import CCUDA

public enum DriverError : UInt32, Error {

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    case invalidValue = 1

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    case outOfMemory = 2

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    case notInitialized = 3

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    case deinitialized = 4

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    case profilerDisabled = 5

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cuProfilerStart or
     * ::cuProfilerStop without initialization.
     */
    @available(*, message: "Deprecated as of CUDA 5.0")
    case profilerNotInitialized = 6

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStart() when profiling is already enabled.
     */
    @available(*, message: "Deprecated as of CUDA 5.0")
    case profilerAlreadyStarted = 7

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cuProfilerStart() when profiling is already enabled.
     */
    @available(*, message: "Deprecated as of CUDA 5.0")
    case profilerAlreadyStopped = 8

    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    case noDevice = 100

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    case invalidDevice = 101

    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    case invalidImage = 200

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    case invalidContext = 201

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    @available(*, message: "Deprecated as of CUDA 3.2")
    case contextAlreadyCurrent = 202

    /**
     * This indicates that a map or register operation has failed.
     */
    case mapFailed = 205

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    case unmapFailed = 206

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    case arrayCurrentlyMapped                = 207

    /**
     * This indicates that the resource is already mapped.
     */
    case alreadyMapped                 = 208

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    case noBinaryForGPU              = 209

    /**
     * This indicates that a resource has already been acquired.
     */
    case alreadyAcquired               = 210

    /**
     * This indicates that a resource is not mapped.
     */
    case notMapped                     = 211

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    case notMappedAsArray            = 212

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    case notMappedAsPointer          = 213

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    case uncorrectableECC              = 214

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    case unsupportedLimit              = 215

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    case contextAlreadyInUse         = 216

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    case peerAccessUnsupported        = 217

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    case invalidPTX                    = 218

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    case invalidGraphicsContext       = 219

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    case nvlinkUncorrectable           = 220

    /**
     * This indicates that the device kernel source is invalid.
     */
    case invalidSource                 = 300

    /**
     * This indicates that the file specified was not found.
     */
    case fileNotFound                 = 301

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    case sharedObjectSymbolNotFound = 302

    /**
     * This indicates that initialization of a shared object failed.
     */
    case sharedObjectInitFailed      = 303

    /**
     * This indicates that an OS call failed.
     */
    case operatingSystem               = 304

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    case invalidHandle                 = 400

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    case symbolNotFound                      = 500

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    case notReady                      = 600

    /**
     * While executing a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    case illegalAddress                = 700

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    case launchOutOfResources        = 701

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    case launchTimeout                 = 702

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    case launchIncompatibleTexturing  = 703

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    case peerAccessAlreadyEnabled    = 704

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     */
    case peerAccessNotEnabled        = 705

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    case primaryContextActive         = 708

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    case contextDestroyed           = 709

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    case assertion                         = 710

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     */
    case tooManyPeers                 = 711

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    case hostMemoryAlreadyRegistered = 712

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    case hostMemoryNotRegistered     = 713

    /**
     * While executing a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    case hardwareStackError           = 714

    /**
     * While executing a kernel, the device encountered an illegal instruction.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    case illegalInstruction            = 715

    /**
     * While executing a kernel, the device encountered a load or store instruction
     * on a memory address which is not aligned.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    case misalignedAddress             = 716

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    case invalidAddressSpace          = 717

    /**
     * While executing a kernel, the device program counter wrapped its address space.
     * The context cannot be used, so it must be destroyed (and a new one should be created).
     * All existing device memory allocations from this context are invalid
     * and must be reconstructed if the program is to continue using CUDA.
     */
    case invalidProgramCounter                     = 718

    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    case launchFailed                  = 719

    /**
     * This error indicates that the attempted operation is not permitted.
     */
    case notPermitted                  = 800

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    case notSupported                  = 801

    /**
     * This indicates that an unknown internal error has occurred.
     */
    case unknown                        = 999


    init(_ cudaError: cudaError_enum) {
        self.init(rawValue: cudaError.rawValue)!
    }

}

func ensureSuccess(_ result: CUresult) throws {
    guard result == CUDA_SUCCESS else {
        throw DriverError(result)
    }
}
