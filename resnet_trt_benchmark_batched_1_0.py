######################################################
# IMPORTS ############################################
######################################################

from multiprocessing import Process, Manager
from PIL import Image
from torchvision import transforms
import multiprocessing as mp
import gc
import time
import random
import pickle as pkl
import torch
import torch_tensorrt
import torchvision
import torch.nn as nn
import os
import subprocess
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity


######################################################
# CONSTANTS & PARAMETERS #############################
######################################################

# Paths
DATA_ROOT = "data"
MODEL_IMAGES_PATH = "./Images"
MODEL_LABELS_PATH = "./Images/labels.txt"

# Device settings
DEVICE = "cuda"

# Dataset configuration
BATCH_SIZE = 16

IMG_SHAPE_V1 = 256
INPUT_V1_PATH = "input_v1.pt"
IMG_SHAPE_V2 = 232
INPUT_V2_PATH = "input_v2.pt"
IMG_CROP = 224
IMG_NORMALIZER_MEAN = 0.485, 0.456, 0.406
IMG_NORMALIZER_STD = 0.229, 0.224, 0.225

NUM_CLASSES = 1000


# Benchmark configuration
MODEL_PYTORCH = "pytorch_"
MODEL_TORCH_COMPILE = "torch_"
MODEL_TRT = "trt_"
NUM_SAMPLES = 1000
NUM_ITERATIONS = 500
WARMUP_ITERATIONS = 500  # Number of warmup iterations if enabled


##################################################
### Funcion Benchmark ############################
##################################################


def benchmark(model_name, images_tensor, labels_tensor, idxs, paths, model_results, model, precision):
    import torch
    from torch.cuda.amp import autocast
    torch.backends.cudnn.benchmark = True
    is_trt_engine = MODEL_TRT in model_name
    model_input_dimm = images_tensor[0].shape
    print(f"Benchmarking {model_name} ({model_input_dimm})...")

    input = images_tensor[0]
    input = input.to(DEVICE)
    
    print("==============\n",
          "Memory before loading in the DEVICE: ", "\n==============")
    mostrar_memoria_cuda()

    ##Execution context
    if is_trt_engine:
        context = model.create_execution_context()
        output_shape = tuple(model.get_binding_shape(1))
        output_tensor = torch.empty(output_shape, dtype=torch.float32, device=DEVICE)
        bindings = [int(input.data_ptr()), int(output_tensor.data_ptr())]
    else:
        model = model.to(DEVICE).eval()
        if MODEL_PYTORCH in model_name:
            model = model.to(memory_format=torch.channels_last)
        
    

    torch.cuda.synchronize()
    print("==============\n", "Memory after loading model into the DEVICE: ", "\n==============")
    mostrar_memoria_cuda()

    # Inicializar los datos
    model_results[model_name]["time"] = 0
    model_results[model_name]["correct"] = 0

    # Warmup iterations
    for _ in range(WARMUP_ITERATIONS):

        idx = _ % len(images_tensor)
        
        input = images_tensor[idx]
        input = input.to(DEVICE)

        if is_trt_engine:
            bindings[0] = int(input.data_ptr())
            context.execute_v2(bindings)
        else:
            with torch.inference_mode(), autocast(dtype=precision):
                _ = model(input)

        torch.cuda.synchronize()
        del input
    print("Warmup completed")

    ##################################

    # Benchmarking iterations with profiling
    print(f"Starting benchmarking for {model_name}...")

        # Bucle sobre los datos
    for i in range(500, NUM_ITERATIONS+WARMUP_ITERATIONS):
        idx = i % len(images_tensor)
        
        # Seleccionar las entradas
        input = images_tensor[idx]
        label = labels_tensor[idx]
        num = idxs[idx]
        path = paths[idx]

        input = input.to(DEVICE)
        label = label.to(DEVICE)

        # with record_function(f"{model_name}_inference"):
            # Guardar el tiempo de inferencia
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if MODEL_PYTORCH in model_name:
            input = input.to(memory_format=torch.channels_last)

        if is_trt_engine:
            start_event.record()
            bindings[0] = int(input.data_ptr())
            context.execute_v2(bindings)
            result = output_tensor.clone()
        else:
            start_event.record()
            with torch.inference_mode(), autocast(dtype=precision):
                
                result = model(input)

        end_event.record()
        torch.cuda.synchronize()
       
            

        # Accumulate timing results
        model_results[model_name]["time"] += start_event.elapsed_time(end_event)

        # Process prediction
        pred = result.argmax(dim=1)


        ####DEBUG
        if i == 0:
            print()
            print("####")
            print("Precison autocast:", precision)
            print("Input shape:", input.shape)
            print("Input dtype:", input.dtype)
            print("####")
            print()
            print("######################################")
            print("RESULTADOS:")
            print("Prediccion: ", pred)
            print("Expected: ", label)
            print("Path: ", path)
            print("IDX: ", num)
            print("######################################")
            print()
            probabilidades = torch.nn.functional.softmax(result, dim=1)
            top5 = torch.topk(probabilidades, 5, dim=1)

            print("Top-5 predicciones:")
            for k in range(5):
                print(f"Clase {top5.indices[0][k].item()}, probabilidad: {top5.values[0][k].item():.4f}")
        ####END_DEBUG



        labels = labels_tensor[idx].to(DEVICE)
        correct = (pred == labels).sum().item() # número de aciertos en el batch dividido entre el tamaño del batch
        model_results[model_name]["correct"] += correct

        # Genera memory leaks, se comprueba imprimiendo por pantalla mejor (arriba)
        # if i == 0:
            # model_results[model_name]["output"] = result.detach().clone()

        del input
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

############################
# LOAD IMAGES ##############
############################

#Carga las imagenes de la carpeta MODEL_IMAGES_PATH con sus labels, las reescala a cierto tamano dado
# y las imagenes que se eligen son las correspondientes a los numeros dados en random_indices.
def cargar_y_procesar_imagenes(tamano, random_indices):
    random_indices = sorted(random_indices)
    #Cargamos los nombres de las imagenes y los labels que les corresponden
    with open(MODEL_LABELS_PATH, 'r') as f:
        all_labels = [int(line.strip()) for line in f.readlines()]

    image_files = sorted([f for f in os.listdir(MODEL_IMAGES_PATH) if f.endswith(".JPEG")])

    assert len(image_files) == len(all_labels)


    #Transformacion necesaria para la entrada de resnet, ver https://docs.pytorch.org/vision/stable/models/resnet.html
    preprocess = transforms.Compose([
        transforms.Resize(tamano),  
        transforms.CenterCrop(IMG_CROP),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=IMG_NORMALIZER_MEAN, std=IMG_NORMALIZER_STD)  
    ])
    
    entradas = []
    total_images = len(image_files)
    primera = True

    for idx in random_indices:
        # Se eligen BATCHSIZE imagenes consecutivas por conveniencia, ajustamos el rango si nos pasamos del total
        if idx + BATCH_SIZE > total_images:
            start = total_images - BATCH_SIZE
            end = total_images
        else:
            start = idx
            end = idx + BATCH_SIZE

        batch_tensors = []
        batch_labels = []
        batch_indices = []
        batch_paths = []

        for i in range(start, end):
            image_path = os.path.join(MODEL_IMAGES_PATH, image_files[i])
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image)
            label = all_labels[i] - 1  # Convertir a 0-indexado

            if primera:
                print("Foto:", i)
                print(image_path, ",", label)
                print()
                primera = False

            batch_tensors.append(image_tensor)
            batch_labels.append(label)
            batch_indices.append(i)
            batch_paths.append(image_path)

        batch_tensor = torch.stack(batch_tensors)  # (BATCH_SIZE, C, H, W)
        batch_labels = torch.tensor(batch_labels)  # (BATCH_SIZE,)

        entradas.append((batch_tensor, batch_labels, batch_indices, batch_paths))

    return entradas


############################################
# MODEL TRACINGS & COMPILING ##############
############################################

def trace_model(name, model, input, precision):


    # Trace model
    tracedmodel = torch.jit.trace(model.cpu(), input.cpu())

    return tracedmodel


def compilar_con_tensorrt(model, dim, precision):
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


    logger = trt.Logger(trt.Logger.INFO)
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


def compilar_con_torch(model):

    return torch.compile(model)

    
def crear_modelo(model_name, models):

    #Elegir el modelo segun el nombre
    if "resnet18" in model_name:
        model = models.resnet18(weights='DEFAULT')
    elif "resnet34" in model_name:
        model = models.resnet34(weights='DEFAULT')
    elif "resnet50" in model_name:
        model = models.resnet50(weights='DEFAULT')
    elif "resnet101" in model_name:
        model = models.resnet101(weights='DEFAULT')
    elif "resnet152" in model_name:
        model = models.resnet152(weights='DEFAULT')

    return model


def crear_modelo_y_benchmark(model_name, images_path, model_results):
    import torch
    import torch_tensorrt
    import torchvision
    import torchvision.models as models
    import torch.nn as nn
    import gc

    #Cargar las entradas de forma correcta
    entradas = torch.load(images_path)

    images_tensor = [x[0] for x in entradas]
    labels_tensor = [x[1] for x in entradas]
    idxs = [x[2] for x in entradas]
    paths = [x[3] for x in entradas]

        
    #En Fp16 si es requerido
    if "fp16" in model_name:
        precision = torch.half
    else:
        precision = torch.float

    model = crear_modelo(model_name, models)


    
    if MODEL_PYTORCH in model_name:
        pass
    elif MODEL_TRT in model_name:
        model = compilar_con_tensorrt(model, tuple(images_tensor[0].shape), precision)
    elif MODEL_TORCH_COMPILE in model_name:
        model = compilar_con_torch(model)

    
    benchmark(model_name, images_tensor, labels_tensor, idxs, paths, model_results, model, precision)

    gc.collect()
    torch._dynamo.reset()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print("Starting benchmark script")


    ############################################################
    # DATA PRELOADING ########################################
    ############################################################

    print(f"Preloading {NUM_ITERATIONS} test samples...")
    
    #Sacamos 1000 indices aleatorios entre 1 y 50000 que son los que vamos a usar para el benchmark
    random_indices = sorted(random.sample(range(50000), WARMUP_ITERATIONS+NUM_ITERATIONS))

    #Los modelos 18 y 34 de resnet usan un tamano y los modelos 50 101 y 152 usan otro
    # queremos igualmente usar las mismas imagenes para todos
    input_v1 = cargar_y_procesar_imagenes(IMG_SHAPE_V1, random_indices)
    torch.save(input_v1, INPUT_V1_PATH)
    input_v2 = cargar_y_procesar_imagenes(IMG_SHAPE_V2, random_indices)
    torch.save(input_v2, INPUT_V2_PATH)

    print(f"Preloaded {len(random_indices)} test samples")

    ########################################################
    # MODELS DECLARATION ###################################
    ########################################################


    models = [
        "model_resnet18",
        "model_resnet34",
        "model_resnet50",
        "model_resnet101",
        "model_resnet152"
    ]



    ############################################################
    # MODELS PREPARATION #######################################
    ############################################################

    # Diccionario con todos los modelos que queremos crear y probar
    models_to_benchmark = []

    #Anadir pytorch models
    for model_name in models:
        models_to_benchmark.append(MODEL_PYTORCH+model_name+"_fp32")
        models_to_benchmark.append(MODEL_PYTORCH+model_name+"_fp16")
        
    #Anadir tensorrt models
    for model_name in models:
        models_to_benchmark.append(MODEL_TRT+model_name+"_fp32")
        models_to_benchmark.append(MODEL_TRT+model_name+"_fp16") 
        
    #Anadir torch models
    for model_name in models:
        models_to_benchmark.append(MODEL_TORCH_COMPILE+model_name+"_fp32")
        models_to_benchmark.append(MODEL_TORCH_COMPILE+model_name+"_fp16") 
        

    

    ############################################################
    # BENCHMARKING LOOP #####################################
    ############################################################

    print(f"Running benchmark with {NUM_ITERATIONS} iterations...")
    torch.cuda.synchronize()  # Ensure previous operations are complete

    # Create directory for profiling traces if it doesn't exist
    os.makedirs("profiler_traces", exist_ok=True)



    manager = Manager()
    model_results = manager.dict()

    print("Numero de modelos: ", len(models_to_benchmark))

    # Mostrarlos todos para debug
    for model_name in models_to_benchmark:

        print("Modelo: ", model_name)

    # Bucle para el benchmark de los modelos
    for model_name in models_to_benchmark:
        
        if "16" or "32" in model_name:
            images_path = INPUT_V1_PATH
        else:
            images_path = INPUT_V2_PATH
        
        print("Creating benchmark for model: ", model_name)

        model_results[model_name] = manager.dict({
            "correct": 0,
            "time": 0.0,
            "output": None
        })
        p = Process(target=crear_modelo_y_benchmark, args=(
            model_name, images_path, model_results))
        p.start()
        p.join()
        torch.cuda.synchronize()


    ############################################################
    # RESULTS REPORTING #####################################
    ############################################################

    print("\n======= BENCHMARK RESULTS =======")
    print(f"Total samples: {NUM_ITERATIONS}\n")
    for model_name in models_to_benchmark:
        print(model_name)


    

    # Calculate and display metrics
    for model_name, results in model_results.items():
        if "time" in results and NUM_ITERATIONS > 0:
            avg_time = results["time"] / NUM_ITERATIONS
            accuracy = ((results["correct"] / NUM_ITERATIONS) / BATCH_SIZE) * 100

            print(f"{model_name.upper()}:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Avg. Inference Time: {avg_time:.3f} ms")


    


    

   

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()