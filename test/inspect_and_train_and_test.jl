using FFTW
using NPZ
using Plots
using NeuralOperators
using Lux, Zygote, Optimisers, Random
using Statistics
using Printf
using Dates

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

# --- 4. Update Helper Functions for 4D Tensors ---
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

mse(a,b) = mean((a .- b).^2)
function loss(ps, st, bx, by)
    ŷ, st2 = model(bx, ps, st)
    # println("st_new structure: ", st2) # Keep for debugging if needed
    return mse(ŷ, by), st2
end

function train_loop!(
    model, 
    ps, 
    st, 
    opt, 
    x_train::Array{Float32,4}, 
    y_train::Array{Float32,4}, 
    n_epochs, 
    batch_size
    )

    metrics_history = Dict(
        "epoch" => Int[],
        "train_time" => Float64[],
        "inferemce_time" => Float64[],  
        "train_mse" => Float64[]
    )
    println("Starting training loop for $n_epochs epochs...")
    
    for epoch in 1:n_epochs
        train_start = time_ns()
        lsum = 0.0
        nb   = 0
        println("Epoch $epoch starting...")
        for (bx, by) in eachbatch(x_train, y_train; bs=batch_size)
            # @info "Input batch shape: $(size(bx))"
            
            # Forward pass
            (ŷ, st_new) = model(bx, ps, st)
            
            # Loss and backward pass
            (l, st_updated), back = Zygote.pullback((ps_inner) -> loss(ps_inner, st_new, bx, by), ps)
            
            gs = back((one(l), nothing))[1]
            
            # Update parameters
            opt, ps = Optimisers.update(opt, ps, gs)
            
            # Update the model state
            st = st_updated
            
            lsum += l
            nb   += 1
        end

        train_time = (time_ns() - train_start) / 1e9
        avg_loss = lsum / nb

        inference_start = time_ns()
        test_mse = evaluate_model(model, ps, st, x_test, y_test, batch_size)
        inference_time = (time_ns() - inference_start) / 1e9

        push!(metrics_history["epoch"], epoch)
        push!(metrics_history["train_time"], train_time)
        push!(metrics_history["inferemce_time"], inference_time)
        push!(metrics_history["train_mse"], avg_loss)

        @printf("Epoch %d/%d:\n", epoch, n_epochs)
        @printf("  Train Time: %.4f s | Inference Time: %.4f s\n", train_time, inference_time)
        @printf("  Train MSE: %.6f\n", avg_loss)

        @printf("  Timestamp: %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        @printf("__________\n")


        println("Epoch $epoch completed. Average MSE: $avg_loss")
        @info "Epoch $epoch | avg MSE = $avg_loss"

        flush(stdout)
    end
    println("Training completed.")
    flush(stdout)
    return ps, st, metrics_history
end

function evaluate_model(model, ps, st, x_test::Array{Float32,4}, y_test::Array{Float32,4}, batch_size)
    lsum = 0.0
    nb   = 0
    for (bx, by) in eachbatch(x_test, y_test; bs=batch_size)
        ŷ, _ = model(bx, ps, st)
        lsum += mse(ŷ, by)
        nb   += 1
    end
    return lsum / nb
end

# --- 5. Train and Evaluate ---
const BATCH_SIZE = 16
const N_EPOCHS = 10
opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)

println("Training for $N_EPOCHS epochs...")
ps, st = train_loop!(model, ps, st, opt, x_train, y_train, N_EPOCHS, BATCH_SIZE)

println("Evaluating on test set…")
test_mse = evaluate_model(model, ps, st, x_test, y_test, BATCH_SIZE)
println("Test MSE = ", test_mse)

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
