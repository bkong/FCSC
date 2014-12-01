function fcsc_fastmulti{T}(X::Array{T, 3}, dictSize::Int, filterSize::Int, optParams::FCSCOpts{T})
# Input:
#   X: HxWxN stack of images (padded already)
#
# Alg. parameters:
#   mu_max: 1e5
#   tau: [1.01 1.5] larger means faster convergence but sacrifices primal feasibility,
#        smaller means slower convergence

  const mu_max = 1e5;
  const tau = 1.01;

  const Dinit    = optParams.Dinit;
  const beta::T  = optParams.beta;
        mu_s::T  = optParams.mu_s;
        mu_t::T  = optParams.mu_t;
  const max_iter = optParams.max_iter;
  const tol      = optParams.tol;
  const DEBUG    = optParams.DEBUG;
  const FIXED_D  = optParams.FIXED_D;



  (convH, convW, numImgs) = size(X);
  convLen = convH*convW;
  assert(!FIXED_D || !isempty(Dinit), "Dinit must be provided when FIXED_D is enabled.");



  # Initialization
  if !FIXED_D;
    mask = ones(T, convH, convW, dictSize);
    mask[1:filterSize, 1:filterSize, :] = 0;
    mask = find(mask);

    if isempty(Dinit);
      Dinit = reshape(mod([convert(T, 1):convH*convW*dictSize], 13), (convH, convW, dictSize));
    else;
      Dinit = padarray(Dinit, (convH-size(Dinit, 1), convW-size(Dinit, 2)), convert(T, 0), "post");
    end
    d = Dinit;
    s = copy(d);
    s[mask] = 0;
    normalize!(s);
    Zt_fft = SharedArray(Complex{T}, (dictSize, convLen));
    ZtZ_fft = SharedArray(Complex{T}, (dictSize, dictSize, convLen));
  else;
    d = padarray(Dinit, (convH-size(Dinit, 1), convW-size(Dinit, 2)), convert(T, 0), "post");
    s = d;
  end
  lambda_s = zeros(T, convH, convW, dictSize);

  # Transform images to the frequency domain
  x_fft = reshape(fft2(X), convLen, numImgs);
  z = zeros(T, convLen*dictSize, numImgs);
  t = zeros(T, convLen*dictSize, numImgs);
  lambda_t = zeros(T, convLen*dictSize, numImgs);
  Dt_fft = zeros(Complex{T}, dictSize, convLen);



  workerIds = workers();
  numWorkers = min(convLen, length(workerIds));
  numPerWorker = int(convLen/numWorkers);
  inds = [{ (i-1)*numPerWorker+(1:numPerWorker) for i=1:numWorkers-1 }; { (numWorkers-1)*numPerWorker+1:convLen }];



  @printf("iter=000");
  if DEBUG;
    debugData = {"x"=>X,
                 "d"=>d, "s"=>s,
                 "lambda_s"=>lambda_s,
                 "mu_s"=>mu_s,
                 "z"=>z, "t"=>t,
                 "lambda_t"=>lambda_t,
                 "mu_t"=>mu_t,
                 "beta"=>beta,
                 "function"=>compute_lagrangian_dictonly};
    (prev_obj,rerr,spar) = compute_lagrangian(debugData);
    @printf(": obj=%.10f (rerr=%.10f, spar=%.10f)\n", prev_obj, rerr, spar);
  end



  iter = 0;
  converged = false;
  if !FIXED_D;
    learnData = LearningData(zeros(Complex{T}, dictSize, convLen), zeros(Complex{T}, dictSize, dictSize, convLen),
      convH, convW, filterSize);
    learnParams = LearningOpts(T); learnParams.mu_s = mu_s; learnParams.max_iter = 1; learnParams.tol = tol;
    if DEBUG;
      learnParams.DEBUG = DEBUG;
      learnParams.debugData = debugData;
    end
  end
  codeData = FastCodingData(zeros(Complex{T}, dictSize, numImgs, convLen), zeros(Complex{T},
    dictSize, dictSize, convLen), convH, convW);
  codeParams = FastCodingOpts(T); codeParams.beta = beta; codeParams.max_iter = 1; codeParams.tol = tol;
    codeParams.t_fft = zeros(Complex{T}, dictSize, numImgs, convLen);
    codeParams.lambda_t_fft = zeros(Complex{T}, dictSize, numImgs, convLen);
  if DEBUG;
    codeParams.DEBUG = DEBUG;
    codeParams.debugData = debugData;
  end
  while ~converged;
    iter += 1;

    @printf("\b\b\b\b\b\b\b\biter=%03d", iter);
    if DEBUG;
      debugData["z"] = z; debugData["t"] = t;
      debugData["lambda_t"] = lambda_t;
      debugData["mu_t"] = mu_t;
      debugData["d"] = d; debugData["s"] = s;
      debugData["lambda_s"] = lambda_s;
      debugData["mu_s"] = mu_s;
      (prev_obj,rerr,spar) = compute_lagrangian(debugData);
      @printf(": obj=%.10f (rerr=%.10f, spar=%.10f)\n", prev_obj, rerr, spar);
    end

    #
    # CODING
    #
    copy!(Dt_fft, reshape(fft2(d), convLen, dictSize)');
    for i=1:convLen;
      # codeData.DtD_fft[:, :, i] = Dt_fft[:, i]*Dt_fft[:, i]';
      codeData.DtD_fft[:, :, i] = broadcast(*, Dt_fft[:, i], Dt_fft[:, i]');
    end
    # NOTE: if numImgs gets too large, we can do this in mini-batches
    for j=1:numImgs;
      # codeData.Dtx_fft[:, j, :] = Dt_fft.*repmat(x_fft[:, j].', dictSize, 1);
      codeData.Dtx_fft[:, j, :] = broadcast(.*, Dt_fft, x_fft[:, j].');
      codeParams.t_fft[:, j, :] = reshape(fft2(reshape(t[:, j], convH, convW, dictSize)), convLen, dictSize).';
      codeParams.lambda_t_fft[:, j, :] = reshape(fft2(reshape(lambda_t[:, j], convH, convW, dictSize)), convLen, dictSize).';
    end
    codeParams.z = z;
    codeParams.t = t;
    codeParams.lambda_t = lambda_t;
    codeParams.mu_t = mu_t;
    (z, t) = fcsc_coding_fastmulti(codeData, codeParams);



    #
    # LEARNING
    #
    if !FIXED_D;
      fill!(learnData.ZtZ_fft, 0);
      fill!(learnData.Ztx_fft, 0);
      for j=1:numImgs;
        copy!(Zt_fft, reshape(fft2(reshape(z[:, j], convH, convW, dictSize)), convLen, dictSize)');
#        for i=1:convLen;
#          # learnData.ZtZ_fft[:, :, i] += Zt_fft[:, i]*Zt_fft[:, i]';
#          learnData.ZtZ_fft[:, :, i] += broadcast(*, Zt_fft[:, i], Zt_fft[:, i]');
#        end
        @sync begin
          for p=1:numWorkers;
            @async begin
              remotecall_wait(workerIds[p], outerprod!, ZtZ_fft, Zt_fft, inds[p]);
            end
          end
        end
        broadcast!(+, learnData.ZtZ_fft, learnData.ZtZ_fft, ZtZ_fft);
        # learnData.Ztx_fft += Zt_fft.*repmat(x_fft[:, j].', dictSize, 1);
        learnData.Ztx_fft += broadcast(.*, Zt_fft, x_fft[:, j].');
      end
      learnParams.Dinit = d;
      learnParams.s = s;
      learnParams.lambda_s = lambda_s;
      learnParams.mu_s = mu_s;
      (d, s) = fcsc_learning_multi(learnData, learnParams);
    end



    # Update the Lagrangian multiplier
    for i=1:convLen*dictSize*numImgs; lambda_t[i] += mu_t*(z[i]-t[i]); end
    if !FIXED_D;
      for i=1:convLen*dictSize; lambda_s[i] += mu_s*(d[i]-s[i]); end
    end
#    debugData["lambda_t"] = lambda_t;
#    debugData["lambda_s"] = lambda_s;
#    (obj,rerr,spar) = compute_lagrangian(debugData);
#    is_mono = (prev_obj>=obj) ? 'O' : 'X';
#    @printf("          L: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
#    prev_obj = obj;

    # Update penalty term
    mu_t = min(tau*mu_t, mu_max);
    mu_s = min(tau*mu_s, mu_max);
#    debugData["mu_t"] = mu_t;
#    debugData["mu_s"] = mu_s;
#    (obj,rerr,spar) = compute_lagrangian(debugData);
#    is_mono = (prev_obj>=obj) ? 'O' : 'X';
#    @printf("          M: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
#    prev_obj = obj;



    learnConverged = FIXED_D || mean((d-s).^2)<=tol;
    codeConverged = true;
    for j=1:numImgs;
      if mu_t>0 && mean((z[:, j]-t[:, j]).^2)>tol;
        codeConverged = false;
        break
      end
    end
    if iter>=max_iter || (learnConverged && codeConverged);
      converged = true;
    end
  end
  s = s[1:filterSize, 1:filterSize, :];

  return (s, z, t);
end
