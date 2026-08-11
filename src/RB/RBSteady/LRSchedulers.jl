abstract type AbstractLRScheduler end

# Helpers

"""
  get_initial_lr(s::AbstractLRScheduler)
Returns the initial learning rate to setup the optimiser.
"""
get_initial_lr(s::AbstractLRScheduler) = error("get_initial_lr not implemented for $(typeof(s))")

"""
  step_scheduler!(s::AbstractLRScheduler, opt_state, epoch::Int, loss)
Updates the learning rate of the optimiser.
"""
step_scheduler!(s::AbstractLRScheduler,opt_state,epoch::Int,total_epochs::Int,loss) = error("step_scheduler! not implemented")

"""
  CosineAnnealing

A learning rate scheduler that implements the Cosine Annealing decay schedule.
It decreases the learning rate from a maximum value
to a minimum value following the shape of a half-cosine wave.

# Fields / Keyword Arguments
- `lr_max::Float32`: The initial, maximum learning rate (default: `0.001f0`).
- `lr_min::Float32`: The final, minimum learning rate (default: `1e-6f0`).
"""
mutable struct CosineAnnealing <: AbstractLRScheduler
  lr_max::Float32
  lr_min::Float32
end

CosineAnnealing(;lr_max=0.001f0,lr_min=1f-6) = CosineAnnealing(lr_max,lr_min)
  
get_initial_lr(s::CosineAnnealing) = s.lr_max

"""
  step_scheduler!(scheduler::CosineAnnealing,opt_state,epoch::Int,_)

Calculates the new learning rate for the current `epoch` using the Cosine Annealing
formula and updates the optimiser state in-place via `Optimisers.adjust!`.

# Arguments
- `scheduler::CosineAnnealing`: The initialized scheduler instance.
- `opt_state`: The current state of the Optimisers.jl optimiser.
- `epoch::Int`: The current training epoch.
- `total_epochs::Int`: Max number of epochs of the training.
- `_`: The current training loss (ignored in this time-based scheduler, but
  required to maintain a unified interface via Multiple Dispatch).
"""
function step_scheduler!(scheduler::CosineAnnealing,opt_state,epoch::Int,total_epochs::Int,_)
  t = min(epoch,total_epochs)
  cos_val = cos(π * (t / total_epochs))
  new_lr = scheduler.lr_min + 0.5f0 * (scheduler.lr_max - scheduler.lr_min) * (1.0f0 + Float32(cos_val))
  
  Optimisers.adjust!(opt_state,new_lr)
end

"""
  ReduceLROnPlateau

Dynamic learning rate scheduler. Reduces the learning rate by a `factor`
when the loss has stopped improving for a given `patience` (number of epochs).
"""
mutable struct ReduceLROnPlateau <: AbstractLRScheduler
  patience::Int
  factor::Float32
  min_lr::Float32
  wait::Int
  best_loss::Float32
  current_lr::Float32
end

ReduceLROnPlateau(;
  patience=100,
  factor=0.5f0,
  min_lr=1f-6,
  start_lr=0.001f0
) = ReduceLROnPlateau(patience,factor,min_lr,0,Inf32,start_lr)

get_initial_lr(s::ReduceLROnPlateau) = s.current_lr

"""
    step_scheduler!(scheduler::ReduceLROnPlateau, opt_state, _, current_loss)

Dispatches the plateau logic. Updates `opt_state`.

# Arguments
- `scheduler::ReduceLROnPlateau`: The initialized scheduler instance.
- `opt_state`: The current state of the Optimisers.jl optimiser.
- `_::Int`: The current training epoch (required to maintain a unified interface).
- `_::Int`: Max number of epochs of the training (required to maintain a unified interface).
- `current_loss`: The current training loss.
"""
function step_scheduler!(scheduler::ReduceLROnPlateau,opt_state,_::Int,_::Int,current_loss)
  if current_loss < scheduler.best_loss
    scheduler.best_loss = current_loss
    scheduler.wait = 0
  else
    scheduler.wait += 1
  end

  if scheduler.wait >= scheduler.patience
    new_lr = max(scheduler.current_lr * scheduler.factor,scheduler.min_lr)
    if new_lr < scheduler.current_lr
      @info "Plateau reached: LR decreased from $(scheduler.current_lr) to $new_lr"
      scheduler.current_lr = new_lr
      Optimisers.adjust!(opt_state,new_lr)
    end
    scheduler.wait = 0
  end
end