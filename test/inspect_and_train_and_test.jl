using FFTW
using NPZ
using Plots
using NeuralOperators
using Lux, Zygote, Optimisers, Random
using Statistics
using Printf
using Dates

# --- Helper Functions ---

function eachbatch(x::Array{Float32,4}, y::Array{Float32,4}; bs::Int)
    n = size(x, 4)  # batch dimension is now the fourth
    batches = Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}()
    i = 1
    while i <= n
        j = min(i + bs - 1, n)
        bx = x[:, :, :, i:j]  # (H, W, channels, batch_slice)
        by = y[:, :, :, i:j]
        push!(batches, (bx, by))
        i = j + 1
    end
    return batches
end

mse(a, b) = mean((a .- b).^2)

function compute_mse(model, ps, st, x::Array{Float32,4}, y::Array{Float32,4}, batch_size::Int)
    """
    Compute MSE over the entire dataset.
    Similar to compute_mse() in PyTorch version.
    """

    start_time_mse = time_ns()
    total_mse = 0.0
    num_batches = 0
    
    for (bx, by) in eachbatch(x, y; bs=batch_size)
        ŷ, _ = model(bx, ps, st)
        total_mse += mse(ŷ, by)
        num_batches += 1
    end
    
    avg_mse = total_mse / num_batches
    elapsed_time_mse = (time_ns() - start_time_mse) / 1e9
    @printf("MSE computation time: %.3f seconds over %d batches\n", elapsed_time_mse, num_batches)
    return avg_mse
end

function measure_inference_time(model, ps, st, x::Array{Float32,4}, y::Array{Float32,4}, 
                                batch_size::Int; warmup_batches::Int=5)
    """
    Measure pure inference time (forward pass only) with warm-up.
    Returns median time per sample in milliseconds and throughput.
    """
    start_time_inference_function = time_ns()
    # Warm-up: discard first measurements for JIT compilation and cache effects
    warmup_count = 0
    for (bx, _) in eachbatch(x, y; bs=batch_size)
        warmup_count += 1
        if warmup_count > warmup_batches
            break
        end
        _, _ = model(bx, ps, st)
    end
    
    # Actual measurement
    batch_times = Float64[]
    
    for (bx, _) in eachbatch(x, y; bs=batch_size)
        t_start = time_ns()
        _, _ = model(bx, ps, st)
        t_end = time_ns()
        
        batch_time_ms = (t_end - t_start) / 1e6  # Convert to milliseconds
        push!(batch_times, batch_time_ms)
    end
    
    # Use median for robustness against outliers
    median_time_per_batch = median(batch_times)
    median_time_per_sample = median_time_per_batch / batch_size
    
    # Calculate throughput (samples/second)
    throughput = (batch_size * 1000.0) / median_time_per_batch

    elapsed_time_inference_function = (time_ns() - start_time_inference_function) / 1e9
    @printf("Inference time measurement completed in %.3f seconds\n", elapsed_time_inference_function)
    
    return median_time_per_sample, throughput
end

function measure_inference_time_multiple_runs(model, ps, st, x::Array{Float32,4}, y::Array{Float32,4},
                                              batch_size::Int; num_runs::Int=5, warmup_batches::Int=5)
    """
    Perform multiple inference time measurements to get robust statistics.
    """
    times_per_sample = Float64[]
    throughputs = Float64[]
    
    for run in 1:num_runs
        time_ms, throughput = measure_inference_time(model, ps, st, x, y, batch_size; 
                                                     warmup_batches=warmup_batches)
        push!(times_per_sample, time_ms)
        push!(throughputs, throughput)
    end
    
    stats = Dict(
        "mean_ms" => mean(times_per_sample),
        "std_ms" => std(times_per_sample),
        "median_ms" => median(times_per_sample),
        "min_ms" => minimum(times_per_sample),
        "max_ms" => maximum(times_per_sample),
        "throughput_mean" => mean(throughputs),
        "throughput_std" => std(throughputs)
    )
    
    return stats
end

# --- 1. Load Data ---
train_file = "./test/train_for_julia.npz"
test_file  = "./test/test_for_julia.npz"

# Load raw data (assuming shape [Batch, H, W])
x_train_raw = convert(Array{Float32}, npzread(train_file)["x"])
y_train_raw = convert(Array{Float32}, npzread(train_file)["y"])
@info "Loaded train raw: x_train size=$(size(x_train_raw)), y_train size=$(size(y_train_raw))"

x_test_raw = convert(Array{Float32}, npzread(test_file)["x"])
y_test_raw = convert(Array{Float32}, npzread(test_file)["y"])
@info "Loaded test raw: x_test size=$(size(x_test_raw)), y_test size=$(size(y_test_raw))"

# --- 2. Reshape Data to (H, W, Channels, Batch) ---
# permutedims(data, (2, 3, 1)) changes (Batch, H, W) -> (H, W, Batch)
# reshape adds the channel dimension: (H, W, Batch) -> (H, W, 1, Batch)

x_train = reshape(permutedims(x_train_raw, (2, 3, 1)), (size(x_train_raw, 2), size(x_train_raw, 3), 1, size(x_train_raw, 1)))
y_train = reshape(permutedims(y_train_raw, (2, 3, 1)), (size(y_train_raw, 2), size(y_train_raw, 3), 1, size(y_train_raw, 1)))
@info "Reshaped train: x_train size=$(size(x_train)), y_train size=$(size(y_train))"

x_test = reshape(permutedims(x_test_raw, (2, 3, 1)), (size(x_test_raw, 2), size(x_test_raw, 3), 1, size(x_test_raw, 1)))
y_test = reshape(permutedims(y_test_raw, (2, 3, 1)), (size(y_test_raw, 2), size(y_test_raw, 3), 1, size(y_test_raw, 1)))
@info "Reshaped test: x_test size=$(size(x_test)), y_test size=$(size(y_test))"

# Get new dimensions
(H_train, W_train, Chan_in, B_train) = size(x_train)
(_,       _,       Chan_out, _)      = size(y_train)
println("Train: H = $H_train, W = $W_train, channels_in = $Chan_in, batch_train = $B_train, channels_out = $Chan_out")

(H_test, W_test, Chan_in2, B_test)   = size(x_test)
(_,      _,       Chan_out2, _)      = size(y_test)
println("Test: H = $H_test, W = $W_test, channels_in = $Chan_in2, batch_test = $B_test, channels_out = $Chan_out2")

@assert Chan_in == Chan_in2 "Input channels mismatch!"
@assert Chan_out == Chan_out2 "Output channels mismatch!"

# --- 3. Define 2D Model ---
rng = Random.default_rng()
# We use 2D modes: (modes_H, modes_W)
model = FourierNeuralOperator(Lux.gelu; chs=(Chan_in, 32, 32, 32, Chan_out), modes=(9, 12))
global ps, st = Lux.setup(rng, model)

println("Model created with $(sum(length, ps)) parameters")

# --- 4. Training Loop ---

# --- 4. Training Loop ---

function loss(ps, st, bx, by)
    ŷ, st2 = model(bx, ps, st)
    return mse(ŷ, by), st2
end

function train_loop!(
    model, 
    ps, 
    st, 
    opt, 
    x_train::Array{Float32,4}, 
    y_train::Array{Float32,4},
    x_test::Array{Float32,4},
    y_test::Array{Float32,4},
    n_epochs::Int, 
    batch_size::Int
    )

    metrics_history = Dict(
        "epoch" => Int[],
        "train_time" => Float64[],
        "eval_time" => Float64[],
        "inference_time_ms" => Float64[],
        "throughput" => Float64[],
        "train_loss" => Float64[],
        "train_mse" => Float64[],
        "test_mse" => Float64[]
    )
    
    println("\n### INICIANDO ENTRENAMIENTO CON MÉTRICAS PERSONALIZADAS ###\n")
    flush(stdout)
    
    for epoch in 1:n_epochs
        epoch_start = time_ns()
        
        # Training phase
        train_loss_epoch = 0.0
        num_batches = 0
        
        for (bx, by) in eachbatch(x_train, y_train; bs=batch_size)
            # Forward pass
            (ŷ, st_new) = model(bx, ps, st)
            
            # Loss and backward pass
            (l, st_updated), back = Zygote.pullback((ps_inner) -> loss(ps_inner, st_new, bx, by), ps)
            
            gs = back((one(l), nothing))[1]
            
            # Update parameters
            opt, ps = Optimisers.update(opt, ps, gs)
            
            # Update the model state
            st = st_updated
            
            train_loss_epoch += l
            num_batches += 1
        end

        train_time = (time_ns() - epoch_start) / 1e9
        avg_train_loss = train_loss_epoch / num_batches
        
        # Evaluation phase
        eval_start = time_ns()
        
        train_mse = compute_mse(model, ps, st, x_train, y_train, batch_size)
        test_mse = compute_mse(model, ps, st, x_test, y_test, batch_size)
        
        eval_time = (time_ns() - eval_start) / 1e9
        
        # Inference time measurement (pure forward pass)
        inference_time_ms, throughput = measure_inference_time(
            model, ps, st, x_test, y_test, batch_size; warmup_batches=5
        )
        
        # Store metrics
        push!(metrics_history["epoch"], epoch)
        push!(metrics_history["train_time"], train_time)
        push!(metrics_history["eval_time"], eval_time)
        push!(metrics_history["inference_time_ms"], inference_time_ms)
        push!(metrics_history["throughput"], throughput)
        push!(metrics_history["train_loss"], avg_train_loss)
        push!(metrics_history["train_mse"], train_mse)
        push!(metrics_history["test_mse"], test_mse)
        
        # Print metrics
        @printf("Epoch %d/%d | Train Loss: %.6f | Train MSE: %.6f | Test MSE: %.6f\n", 
                epoch, n_epochs, avg_train_loss, train_mse, test_mse)
        @printf("Train Time: %.2fs | Eval Time: %.2fs | Inference: %.3fms/sample (%.1f samples/s)\n",
                train_time, eval_time, inference_time_ms, throughput)
        @printf("Timestamp: %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println("_" ^ 80)
        flush(stdout)
    end
    
    println("\n### ENTRENAMIENTO COMPLETADO ###\n")
    flush(stdout)
    
    return ps, st, metrics_history
end

# --- 5. Train and Evaluate ---
const BATCH_SIZE = 32
const N_EPOCHS = 10
opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)

println("Training for $N_EPOCHS epochs with batch size $BATCH_SIZE...")
ps, st, metrics_history = train_loop!(model, ps, st, opt, x_train, y_train, x_test, y_test, N_EPOCHS, BATCH_SIZE)

# Final inference time measurement with multiple runs
println("\n### MEDICIÓN FINAL DE INFERENCE TIME (5 runs) ###\n")
final_stats = measure_inference_time_multiple_runs(model, ps, st, x_test, y_test, BATCH_SIZE; 
                                                   num_runs=5, warmup_batches=5)

@printf("Inference Time Statistics (5 runs):\n")
@printf("  Mean: %.3f ± %.3f ms/sample\n", final_stats["mean_ms"], final_stats["std_ms"])
@printf("  Median: %.3f ms/sample\n", final_stats["median_ms"])
@printf("  Range: [%.3f, %.3f] ms/sample\n", final_stats["min_ms"], final_stats["max_ms"])
@printf("  Throughput: %.1f ± %.1f samples/s\n", final_stats["throughput_mean"], final_stats["throughput_std"])
flush(stdout)

# Save metrics to CSV
println("\nSaving metrics to CSV...")
open("metrics_history_julia.csv", "w") do f
    write(f, "epoch,train_time,eval_time,inference_time_ms,throughput,train_loss,train_mse,test_mse\n")
    for i in 1:length(metrics_history["epoch"])
        write(f, @sprintf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                         metrics_history["epoch"][i],
                         metrics_history["train_time"][i],
                         metrics_history["eval_time"][i],
                         metrics_history["inference_time_ms"][i],
                         metrics_history["throughput"][i],
                         metrics_history["train_loss"][i],
                         metrics_history["train_mse"][i],
                         metrics_history["test_mse"][i]))
    end
end

# Save final inference stats
open("inference_stats_final_julia.txt", "w") do f
    write(f, "=== Final Inference Time Statistics (5 runs) ===\n")
    write(f, @sprintf("Mean: %.6f ms/sample\n", final_stats["mean_ms"]))
    write(f, @sprintf("Std Dev: %.6f ms/sample\n", final_stats["std_ms"]))
    write(f, @sprintf("Median: %.6f ms/sample\n", final_stats["median_ms"]))
    write(f, @sprintf("Min: %.6f ms/sample\n", final_stats["min_ms"]))
    write(f, @sprintf("Max: %.6f ms/sample\n", final_stats["max_ms"]))
    write(f, @sprintf("Throughput Mean: %.2f samples/s\n", final_stats["throughput_mean"]))
    write(f, @sprintf("Throughput Std: %.2f samples/s\n", final_stats["throughput_std"]))
end

println("Metrics saved to 'metrics_history_julia.csv' and 'inference_stats_final_julia.txt'")
flush(stdout)

# --- 6. Update Plotting to 2D Heatmap ---
println("Plotting test sample...")
i = 1
x1 = x_test[:, :, :, i:i] # (H, W, 1, 1)
y1 = y_test[:, :, :, i:i]
ŷ1, _ = model(x1, ps, st)

# Create two heatmap plots
p1 = heatmap(y1[:, :, 1, 1], title="Ground Truth (y)", aspect_ratio=:equal, c=:viridis)
p2 = heatmap(ŷ1[:, :, 1, 1], title="FNO Prediction (ŷ)", aspect_ratio=:equal, c=:viridis)

# Combine them side-by-side
plot(p1, p2, layout=(1, 2), size=(800, 400))
title!("True vs Predicted – Test sample $i")
savefig("true_vs_pred_test_sample_2D.png")

println("Saved plot: true_vs_pred_test_sample_2D.png")
println("Finished.")
