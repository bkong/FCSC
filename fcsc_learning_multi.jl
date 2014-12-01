# Helper functions
function d_subproblem{T}(s_fft::Array{Complex{T}, 2}, mu_s::T, lambda_s_fft::Array{Complex{T}, 2},
                         ZtZ_fft::Array{Complex{T}, 3}, Ztx_fft::Array{Complex{T}, 2})
  convLen = size(s_fft, 2);
  dictSize = size(ZtZ_fft, 1);
  # (Eq. 14)
  d_fft = zeros(s_fft);
  A = zeros(Complex128, dictSize, dictSize); b = zeros(Complex128, dictSize, 1);
  for i=1:convLen;
    # d_fft[:, i] = ( ZtZ_fft[:, :, i] + mu_s.*I ) \ ( Ztx_fft[:, i] + mu_s.*s_fft[:, i] - lambda_s_fft[:, i] );
    copy!(A, ZtZ_fft[:, :, i]);
    for j=1:dictSize; A[j, j] += mu_s; end
    copy!(b, s_fft[:, i]);
    scale!(b, mu_s);
    b += Ztx_fft[:, i];
    b -= lambda_s_fft[:, i];
    d_fft[:, i] = A\b;
  end
  return d_fft::Array{Complex{T}, 2};
end

function s_subproblem{T}(d::Array{T, 3}, mu_s::T, lambda_s::Array{T, 3}, mask::Array{Int, 1})
  # (Eq. 20)
  # s = d + 1/mu_s.*lambda_s;
  s = copy(lambda_s);
  scale!(s, 1/mu_s);
  s += d;
  s[mask] = 0;
  # (Eq. 18)
  normalize!(s);
  return s::Array{T, 3};
end


function fcsc_learning_multi{T}(data::LearningData{T}, optParams::LearningOpts{T})
# Alg. parameters:
#   mu_max: 1e5
#   tau: [1.01 1.5] larger means faster convergence but sacrifices primal feasibility,
#        smaller means slower convergence

  const mu_max = 1e5;
  const tau = 1.01;

  const ZtZ_fft    = data.ZtZ_fft;
  const Ztx_fft    = data.Ztx_fft;
  const convH      = data.convH;
  const convW      = data.convW;
  const filterSize = data.filterSize;
  const d          = optParams.Dinit;
        s          = optParams.s;
        lambda_s   = optParams.lambda_s;
        mu_s::T    = optParams.mu_s;
  const max_iter   = optParams.max_iter;
  const tol        = optParams.tol;
  const DEBUG      = optParams.DEBUG;
        debugData  = optParams.debugData;



  dictSize = size(ZtZ_fft, 1);
  convLen = convH*convW;

  assert(size(ZtZ_fft, 3)==convLen, "ZtZ_fft does not have the correct depth.")
  assert(size(Ztx_fft, 1)==dictSize && size(Ztx_fft, 2)==convLen,
    "Ztx_fft does not have the correct dimensions.")
  assert(isempty(d) || size(d, 3)==dictSize,
    "Number of filters in Dinit does not match the number of channels in Z.")


  mask = ones(T, convH, convW, dictSize);
  mask[1:filterSize, 1:filterSize, :] = 0;
  mask = find(mask);


  # Initialization
  if isempty(lambda_s);
    lambda_s = zeros(T, convH, convW, dictSize);
  end
  lambda_s_fft = reshape(fft2(lambda_s), convLen, dictSize).';
  if isempty(d);
    d = zeros(T, convH, convW, dictSize);
  end
  d_fft = reshape(fft2(d), convLen, dictSize).';
  if isempty(s);
    s = s_subproblem(d, mu_s, lambda_s, mask);
  end
  s_fft = reshape(fft2(s), convLen, dictSize).';



  if DEBUG;
    debugData["d"] = d;
    debugData["s"] = s;
    debugData["lambda_s"] = lambda_s;
    (prev_obj, rerr, spar) = compute_lagrangian(debugData);
  end



  iter = 0;
  converged = false;
  while ~converged;
    iter += 1;

    if max_iter>1; @printf("\b\b\b%03d", iter); end

    # Solve for d* in the frequency domain
    d_fft = d_subproblem(s_fft, mu_s, lambda_s_fft, ZtZ_fft, Ztx_fft);
    d = real(ifft2(reshape(d_fft.', convH, convW, dictSize)));
    if DEBUG;
      debugData["d"] = d;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          D: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end


    # Solve for s* in the spatial domain
    s = s_subproblem(d, mu_s, lambda_s, mask);
    if DEBUG;
      debugData["s"] = s;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          S: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end


    if max_iter==1; return (d, s); end # Don't do unnecessary work
    s_fft = reshape(fft2(s), convLen, dictSize).';


    # Update the Lagrangian multipliers
    for i=1:convLen*dictSize; lambda_s[i] += mu_s*( d[i]-s[i] ); end
    lambda_s_fft = reshape(fft2(lambda_s), convLen, dictSize).';
    if DEBUG;
      debugData["lambda_s"] = lambda_s;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          L: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end

    # Update penalty term
    mu_s = min(tau*mu_s, mu_max);
    if DEBUG;
      debugData["mu_s"] = mu_s;
      (obj, rerr, spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          M: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end



    if iter>=max_iter || mean((d-s).^2)<=tol;
      converged = true;
    end
  end

  return (d, s);

end
