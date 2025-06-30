######################################################
# IMPORTS ############################################
######################################################

from multiprocessing import Process, Manager
import multiprocessing as mp
import gc
import time
import pickle as pkl
import torch
import torch_tensorrt
import torchvision
import torch.nn as nn
import os
import subprocess


from torch.profiler import profile, record_function, ProfilerActivity


######################################################
# CONSTANTS & PARAMETERS #############################
######################################################

# Paths
DATA_ROOT = "data"
MODEL_PARAMS_PATH = "fasionmnist_mlp_params.pkl"

# Device settings
DEVICE = "cuda"

# Dataset configuration
BATCH_SIZE = 8192
FEATURE_DIM_28 = 784  # 28x28 flattened
FEATURE_DIM_128 = 16384  # 128x128 flattened
FEATURE_DIM_512 = 262144  # 512x512 flattened
FEATURE_DIM_2048 = 4194304  # 2048x2048 flattened
FEATURE_DIM_PRUEBAS = 1024  # Para las pruebas con nodos intermedios variables

IMG_SHAPE = (1, 28, 28)
NUM_CLASSES = 10

# Dimensiones de los nodos intermedios
FEATURED_NODES_DIM_1GB = 256000
FEATURED_NODES_DIM_2GB = 512000
FEATURED_NODES_DIM_4GB = 1000000
FEATURED_NODES_DIM_8GB = 2000000

FEATURED_DIM_OUTPUT = 32

# Type of compilers

TRT_BASE_COMPILER = "trt_original_"
TRT_TORCH_COMPILER = "trt_pytorchf_"
PYTORCH = "pytorch_"
TORCH_COMPILER = "torch_compiler_"

# TensorRT configuration
TRT_MIN_BATCH = 1
TRT_OPT_BATCH = 8
TRT_MAX_BATCH = 32
TRT_INPUT_SHAPE_28 = [FEATURE_DIM_28]
TRT_INPUT_SHAPE_128 = [FEATURE_DIM_128]
TRT_INPUT_SHAPE_512 = [FEATURE_DIM_512]
TRT_INPUT_SHAPE_2048 = [FEATURE_DIM_2048]

# Benchmark configuration
NUM_ITERATIONS = 500
WARMUP_ITERATIONS = 500  # Number of warmup iterations if enabled

# Class names for interpretation
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

##################################################
### Funcion Benchmark ############################
##################################################


def benchmark(model_name, model_info, test_samples, model_results, model, precision):
    import torch
    from torch.cuda.amp import autocast
    torch.backends.cudnn.benchmark = True
    model_input_dimm = model_info["input_dimm"]
    print(f"Benchmarking {model_name} ({model_input_dimm})...")
    print("==============\n", "Memory before loading model: ", "\n==============")
    mostrar_memoria_cuda()

    
    is_trt_engine = TRT_BASE_COMPILER in model_name

    

    # move it to GPU
    print("==============\n",
          "Memory before loading in the DEVICE: ", "\n==============")
    mostrar_memoria_cuda()
    
    

    torch.cuda.synchronize()
    print("==============\n", "Memory after loading model: ", "\n==============")
    mostrar_memoria_cuda()

    # Initialize timing and accuracy
    model_results[model_name]["time"] = 0
    model_results[model_name]["correct"] = 0

    # Warmup iterations (use test_samples and go cycling them)
    for _ in range(WARMUP_ITERATIONS):
        idx = _ % len(test_samples)
        (img32, img_fp16, label_cuda) = test_samples[idx]

        #Creación de las entradas falsas
        if "fp16" in model_name and (TORCH_COMPILER in model_name or PYTORCH in model_name):
            inputs = img_fp16
        else:
            inputs = img32
        
        if "128" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_128)
        elif "512" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_512)
        elif "2048" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_2048)
        elif "GB" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_PRUEBAS)

        if "fp16" in model_name and (TORCH_COMPILER in model_name or PYTORCH in model_name) and not "base" in model_name:
            inputs = inputs.half()
        

        
        inputs = inputs.to(DEVICE)
        #Modo Inferencia
        if is_trt_engine:
            context = model.create_execution_context()
            output_shape = tuple(model.get_binding_shape(1))
            output_tensor = torch.empty(output_shape, dtype=torch.float32, device=DEVICE)
            bindings = [int(inputs.data_ptr()), int(output_tensor.data_ptr())]
        else:
            model.eval()
            model.to(DEVICE)

            #Info debug
            if idx == 1:
                print("Input shape:", inputs.shape)
                print("Input dtype:", inputs.dtype)

        #Inferencia
        with autocast(dtype=precision), torch.inference_mode():
            if is_trt_engine:
                bindings[0] = int(inputs.data_ptr())
                context.execute_v2(bindings)
            else:
                _ = model(inputs)
            torch.cuda.synchronize()
        del inputs
    print("Warmup completed")

    ##################################

    # Benchmarking iterations with profiling
    print(f"Starting benchmarking for {model_name}...")

  
   
    # Loop over test samples
    for i in range(NUM_ITERATIONS):
        idx = i % len(test_samples)
        (img32, img_fp16, label_cuda) = test_samples[idx]

        if "fp16" in model_name and (TORCH_COMPILER in model_name or PYTORCH in model_name):
            inputs = img_fp16
        else:
            inputs = img32
        
        if "128" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_128)
        elif "512" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_512)
        elif "2048" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_2048)
        elif "GB" in model_name:
            inputs = torch.rand(BATCH_SIZE, FEATURE_DIM_PRUEBAS)

        if "fp16" in model_name and (TORCH_COMPILER in model_name or PYTORCH in model_name) and not "base" in model_name:
            inputs = inputs.half()
        



        inputs = inputs.to(DEVICE)
        if i == 1:
            print("Input shape:", inputs.shape)
            print("Input dtype:", inputs.dtype)
        # Record function to label this section in the trace
        with autocast(dtype=precision), torch.inference_mode():
            # time the inference for the given sample
            torch.cuda.synchronize()  # Ensure GPU is synchronized before timing
            time_start = time.time()

            if is_trt_engine:
                bindings[0] = int(inputs.data_ptr())
                context.execute_v2(bindings)
                result = output_tensor.clone()
            else:
                result = model(inputs)
            torch.cuda.synchronize()  # Ensure inference is complete before stopping timer
            time_end = time.time()

        # Accumulate timing results
        model_results[model_name]["time"] += time_end - time_start

        # Process prediction
        if is_trt_engine and BATCH_SIZE == 1:
                pred = result.argmax()
                if pred.to(DEVICE) == label_cuda:
                    model_results[model_name]["correct"] += 1
        else:
            pred = result.argmax(dim=-1)            
            correct_count = (pred == label_cuda).sum().item()
            model_results[model_name]["correct"] += correct_count



        del inputs
        del result
        del pred

    

    # Memory liberation
    del (model)
    gc.collect()
    torch.cuda.synchronize()

##################################################
### Funcion para ver la memoria de la grafica ####
###################################################

def mostrar_memoria_cuda():

    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,nounits,noheader"],
        encoding='utf-8'
    )

    gpus = output.strip().split('\n')

    for i, gpu in enumerate(gpus):
        total, used, free = map(int, gpu.split(','))
        print(f"GPU {i}:")
        print(f"  Total memory: {total} MB")
        print(f"  Used memory:  {used} MB")
        print(f"  Free memory:  {free} MB")

#########################################################
# MODEL DEFINITION w/ PRE-TRAINED WEIGHTS ##############
#########################################################

class MLPModel(nn.Module):
    def __init__(self, w0, b0, w1, b1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(w0.shape[1], w0.shape[0])
        self.fc2 = nn.Linear(w1.shape[1], w1.shape[0])
        self.relu = nn.ReLU()
       
        self.fc1.weight = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.fc1.bias = nn.Parameter(torch.tensor(b0, dtype=torch.float32))
        self.fc2.weight = nn.Parameter(torch.tensor(w1, dtype=torch.float32))
        self.fc2.bias = nn.Parameter(torch.tensor(b1, dtype=torch.float32))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

############################################
# MODEL TRACINGS & COMPILING ##############
############################################

#Codigo auxiliar para añadir los modelos a la estructura de datos
def anadir_modelos(models, models_to_benchmark, model_type):

    for model_name, input, dim in models:
        if "GB" in model_name:
            nodes_dimm = FEATURED_NODES_DIM_1GB
            if "2" in model_name:
                nodes_dimm = FEATURED_NODES_DIM_2GB
            elif "4" in model_name:
                nodes_dimm = FEATURED_NODES_DIM_4GB
            elif "8" in model_name:
                nodes_dimm = FEATURED_NODES_DIM_8GB

            models_to_benchmark[model_type+model_name+"_fp32"] = {"input": input, "input_dimm": dim, "nodes_dimm": nodes_dimm, "output_dimm": FEATURED_DIM_OUTPUT}
            models_to_benchmark[model_type+model_name+"_fp16"] = {"input": input, "input_dimm": dim, "nodes_dimm": nodes_dimm, "output_dimm": FEATURED_DIM_OUTPUT}
        else:
            models_to_benchmark[model_type+model_name+"_fp32"] = {"input": input, "input_dimm": dim, "nodes_dimm": None, "output_dimm": None}
            models_to_benchmark[model_type+model_name+"_fp16"] = {"input": input, "input_dimm": dim, "nodes_dimm": None, "output_dimm": None}


def trace_model(name, model, input, dim, precision):

    #input = input.reshape(1, dim).float()
    input = input.view(input.size(0), -1)  
    if TORCH_COMPILER in name and "fp16" in name:
        input = input.half()

    # Prepare tracing input and load into GPU
    print(f"Tracing model: {name}")
    print(f" - input shape before reshape: {input.shape}")
    print(f" - input numel: {input.numel()}, expected: {BATCH_SIZE * dim}")
    print(f" - input type: {input.dtype}")

    
    # Trace model
    tracedmodel = torch.jit.trace(model, input)

    return tracedmodel

def compilar_con_tensorrt_base(model, dim, precision):
    import tensorrt as trt


    onnx_path = "model_temp.onnx"
    dummy_input = torch.randn(dim).cpu()

    model.eval().cpu()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )


    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Error al parsear ONNX")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if precision == torch.half:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    assert engine is not None, "Fallo al construir el motor de TensorRT"

    return engine
    

def compilar_con_tensorrt(model, dim, precision):
    print("########################")
    print("Memoria Pre Compilar")
    print("########################")
    mostrar_memoria_cuda()

    #Compile with tensorrt
    return torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(
            shape=[BATCH_SIZE, dim],
            dtype=torch.float
        )],
        enabled_precisions={precision}
    )

def compilar_con_torch(model):

    #Compile with torch
    return torch.compile(model)
    
def crear_modelo(input_dimm, nodes_dimm, output_dimm, mlp_params):
    
    if input_dimm != FEATURE_DIM_28:
        if output_dimm and nodes_dimm:
            w0 = torch.rand(nodes_dimm, input_dimm)
            b0 = torch.rand(nodes_dimm)
            w1 = torch.rand(32, nodes_dimm)
            b1 = torch.rand(output_dimm)
        else:
            w0 = torch.rand(128, input_dimm)
            b0 = mlp_params["b0"]
            w1 = mlp_params["w1"]
            b1 = mlp_params["b1"]
    else:
        w0 = mlp_params["w0"]
        b0 = mlp_params["b0"]
        w1 = mlp_params["w1"]
        b1 = mlp_params["b1"]   

    return MLPModel(w0,b0,w1,b1)

#main del hilo que crea el modelo y hace el benchmark
def crear_modelo_y_benchmark(model_name, model_info, test_samples, model_results, mlp_params):
    import torch
    import torch_tensorrt
    import torchvision
    import torch.nn as nn
    import gc

    MODEL_PARAMS_PATH = "fasionmnist_mlp_params.pkl"


    mlp_params = pkl.load(open(MODEL_PARAMS_PATH, "rb"))

    
    input = model_info["input"]
    input_dimm = model_info["input_dimm"]
    nodes_dimm = model_info["nodes_dimm"]
    output_dimm = model_info["output_dimm"]

    print("###################")
    print("Mostrar memoria antes de crear el modelo")
    mostrar_memoria_cuda()
    print("###################")


    model = crear_modelo(input_dimm, nodes_dimm, output_dimm, mlp_params)

    if "fp16" in model_name:
        precision = torch.half
        if PYTORCH in model_name or TORCH_COMPILER in model_name:
            model = model.half()
    else:
        precision = torch.float
    
    print("###################")
    print("Mostrar memoria despues de crear el modelo")
    mostrar_memoria_cuda()
    print("###################")

    if TRT_TORCH_COMPILER in model_name or TORCH_COMPILER in model_name:
        model = trace_model (model_name, model, input, input_dimm, precision)
    
        
    
    print("###################")
    print("Mostrar memoria despues de tracear el modelo")
    mostrar_memoria_cuda()
    print("###################")
    
    if PYTORCH in model_name:
        pass
    elif TRT_TORCH_COMPILER in model_name:
        model = compilar_con_tensorrt(model, input_dimm, precision)
    elif TORCH_COMPILER in model_name:
        model = compilar_con_torch(model)
    elif TRT_BASE_COMPILER in model_name:
        model = compilar_con_tensorrt_base(model, input_dimm, precision)

   
    
    benchmark(model_name, model_info, test_samples, model_results, model, precision)

    gc.collect()
    torch._dynamo.reset()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


def main():

    gc.collect()
    torch._dynamo.reset()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()

    print("Starting benchmark script")
    mostrar_memoria_cuda()
    ########################################################
    # DATASET LOADING & INPUTS##############################
    ########################################################


    test_data = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    img, label = next(iter(test_loader))
    img = img.view(img.size(0), -1)  
 
    print(f"Input image dtype: {img.dtype}")


    mlp_params = pkl.load(open(MODEL_PARAMS_PATH, "rb"))
    
    input_128 = torch.rand(1, FEATURE_DIM_128)
    input_512 = torch.rand(1, FEATURE_DIM_512)
    input_2048 = torch.rand(1, FEATURE_DIM_2048)
    input_Pruebas = torch.rand(1, FEATURE_DIM_PRUEBAS)


    ########################################################
    # MODELS DECLARATION ###################################
    ########################################################


    models = [
        ("model_base", img, FEATURE_DIM_28),
        ("model_128", input_128, FEATURE_DIM_128),
        ("model_512", input_512, FEATURE_DIM_512),
        ("model_2048", input_2048, FEATURE_DIM_2048),
        ("model_1GB", input_Pruebas, FEATURE_DIM_PRUEBAS),
        ("model_2GB", input_Pruebas, FEATURE_DIM_PRUEBAS),
        ("model_4GB", input_Pruebas, FEATURE_DIM_PRUEBAS)
        ("model_8GB", input_Pruebas, FEATURE_DIM_PRUEBAS)
    ]



    ############################################################
    # MODELS PREPARATION #######################################
    ############################################################

    # Diccionario con todos los modelos que queremos crear y probar
    models_to_benchmark = {
    }
    #Anadir pytorch models
    anadir_modelos(models, models_to_benchmark, PYTORCH)
        
    #Anadir tensorrt_torch models
    anadir_modelos(models, models_to_benchmark, TRT_TORCH_COMPILER)
        
    #Anadir torch models
    anadir_modelos(models, models_to_benchmark, TORCH_COMPILER)

    #Anadir trt origin models
    anadir_modelos(models, models_to_benchmark, TRT_BASE_COMPILER)

    # Mostrarlos todos para debug
    for model_name, model_info in models_to_benchmark.items():
        input_tensor = model_info["input"]
        dim = model_info["input_dimm"]
        nodes_dimm = model_info["nodes_dimm"]
        output_dimm = model_info["output_dimm"]
        print(model_name, ": ", "{input: ", input, ", dim: ", dim,"}")


    ############################################################
    # DATA PRELOADING ########################################
    ############################################################

    print(f"Preloading {NUM_ITERATIONS} test samples...")
    test_samples = []
    test_loader_iter = iter(test_loader)

    # Preload all data samples to ensure consistent comparison
    for _ in range(NUM_ITERATIONS):
        try:
            img, label = next(test_loader_iter)
        except StopIteration:
            test_loader_iter = iter(test_loader)
            img, label = next(test_loader_iter)

        # Mueve a device
        label_cuda = label.to(DEVICE)

        # Aplana imagen para el modelo
        img32 = img.view(img.size(0), -1)       
        img_fp16 = img32.half()  


        # Los ponemos en una estructura de datos
        test_samples.append((img32, img_fp16,label_cuda))


    print(f"Preloaded {len(test_samples)} test samples")


    ############################################################
    # BENCHMARKING LOOP #####################################
    ############################################################

    print(f"Running benchmark with {NUM_ITERATIONS} iterations...")
    torch.cuda.synchronize()  # Ensure previous operations are complete

    # Create directory for profiling traces if it doesn't exist
    os.makedirs("profiler_traces", exist_ok=True)

    # Loop over the models
  
 
    manager = Manager()
    model_results = manager.dict()

    for key in models_to_benchmark.keys():
        
        model_info = models_to_benchmark[key]

        dim = model_info["input_dimm"]

        print("Creating benchmark for model ",key, "( dim: ", dim, ")")

        model_results[key] = manager.dict({
            "correct": 0,
            "time": 0.0,
            "output": None
        })
        p = Process(target=crear_modelo_y_benchmark, args=(
            key, model_info, test_samples, model_results, mlp_params))
        p.start()
        p.join()
        p.close()
        del p
        torch.cuda.synchronize()


    ############################################################
    # RESULTS REPORTING #####################################
    ############################################################

    print("\n======= BENCHMARK RESULTS =======")
    print(f"Total samples: {NUM_ITERATIONS}\n")
    for model_name, modelinfo in models_to_benchmark.items():
        print(model_name)



    # Calculate and display metrics
    for model_name, results in model_results.items():
        if "time" in results and NUM_ITERATIONS > 0:
            avg_time = results["time"] / NUM_ITERATIONS
            accuracy = (results["correct"] / (NUM_ITERATIONS * BATCH_SIZE)) * 100

            print(f"{model_name.upper()}:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Avg. Inference Time: {avg_time*1000:.3f} ms")




if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()