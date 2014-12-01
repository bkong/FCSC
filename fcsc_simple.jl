function fcsc_simple{T}(X::Array{T, 3}, dictSize::Int, filterSize::Int, optParams::FCSCOpts{T})
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
  mask = ones(T, convH, convW, dictSize);
  mask[1:filterSize, 1:filterSize, :] = 0;
  mask = find(mask);

  if !FIXED_D;
    if isempty(Dinit);
      Dinit = reshape(mod([convert(T, 1):convH*convW*dictSize], 13), (convH, convW, dictSize));
    else;
      Dinit = padarray(Dinit, (convH-size(Dinit, 1), convW-size(Dinit, 2)), convert(T, 0), "post");
    end
    d = Dinit;
    s = copy(d);
    s[mask] = 0;
    normalize!(s);
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



  d_fft = reshape(fft2(d), convLen, dictSize);
  D_fft = zeros(Complex{T}, convLen, convLen*dictSize);
  for k=1:dictSize;
    D_fft[:, (k-1)*convLen+[1:convLen]] = diagm(d_fft[:, k]);
  end
  DtD_fft = D_fft'*D_fft;
  for j=1:numImgs;
    #
    # subproblem z
    Dtx_fft = D_fft'*x_fft[:, j];

    # Initialize z to pure reconstruction solution
    z_fft = ( DtD_fft + mu_t.*eye(convLen*dictSize) ) \ Dtx_fft;
    z[:, j] = vec(real(ifft2(reshape(z_fft, convH, convW, dictSize))));

    #
    # subproblem t
    t[:, j] = shrink(z[:, j], beta/mu_t);
  end



  iter = 0;
  converged = false;
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
    d_fft = reshape(fft2(d), convLen, dictSize);
    D_fft = zeros(Complex{T}, convLen, convLen*dictSize);
    for k=1:dictSize;
      D_fft[:, (k-1)*convLen+[1:convLen]] = diagm(d_fft[:, k]);
    end
    DtD_fft = D_fft'*D_fft;
    for j=1:numImgs;
      #
      # subproblem z
      Dtx_fft = D_fft'*x_fft[:, j];
      t_fft = vec(fft2(reshape(t[:, j], convH, convW, dictSize)));
      lambda_t_fft = vec(fft2(reshape(lambda_t[:, j], convH, convW, dictSize)));

      z_fft = ( DtD_fft + mu_t.*eye(convLen*dictSize) ) \ ( Dtx_fft + mu_t.*t_fft - lambda_t_fft );
      z[:, j] = vec(real(ifft2(reshape(z_fft, convH, convW, dictSize))));

      #
      # subproblem t
      t[:, j] = shrink(z[:, j]+lambda_t[:, j]./mu_t, beta/mu_t);
    end
    if DEBUG;
      debugData["z"] = z;
      debugData["t"] = t;
      (obj,rerr,spar) = compute_lagrangian(debugData);
      is_mono = (prev_obj>=obj) ? 'O' : 'X';
      @printf("          C: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
      prev_obj = obj;
    end



    #
    # LEARNING
    #
    if !FIXED_D;
      #
      # subproblem d
      Z_fft = zeros(Complex{T}, convLen, convLen*dictSize);
      ZtZ_fft = zeros(Complex{T}, convLen*dictSize, convLen*dictSize);
      Ztx_fft = zeros(Complex{T}, convLen*dictSize);
      for j=1:numImgs;
        z_fft = reshape(fft2(reshape(z[:, j], convH, convW, dictSize)), convLen, dictSize);
        for k=1:dictSize;
          Z_fft[:, (k-1)*convLen+[1:convLen]] += diagm(z_fft[:, k]);
        end

        ZtZ_fft += Z_fft'*Z_fft;
        Ztx_fft += Z_fft'*x_fft[:, j];
      end
      s_fft = vec(fft2(s));
      lambda_s_fft = vec(fft2(lambda_s));
      d_fft = ( ZtZ_fft + mu_s.*eye(convLen*dictSize) ) \ ( Ztx_fft + mu_s.*s_fft - lambda_s_fft );
      d = real(ifft2(reshape(d_fft, convH, convW, dictSize)));
      if DEBUG;
        debugData["d"] = d;
        (obj,rerr,spar) = compute_lagrangian(debugData);
        is_mono = (prev_obj>=obj) ? 'O' : 'X';
        @printf("          D: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
        prev_obj = obj;
      end

      #
      # subproblem s
      s = d + lambda_s/mu_s;
      s[mask] = 0;
      normalize!(s);
      if DEBUG;
        debugData["s"] = s;
        (obj,rerr,spar) = compute_lagrangian(debugData);
        is_mono = (prev_obj>=obj) ? 'O' : 'X';
        @printf("          S: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
        prev_obj = obj;
      end
    end



    #
    # update lambda multipliers
    for j=1:numImgs;
      for i=1:convLen*dictSize;
        lambda_t[i, j] += mu_t*( z[i, j]-t[i, j] );
      end
    end
    for i=1:convLen*dictSize;
      lambda_s[i] += mu_s*( d[i]-s[i] );
    end
    if DEBUG;
#      debugData["lambda_t"] = lambda_t;
#      debugData["lambda_s"] = lambda_s;
#      (obj,rerr,spar) = compute_lagrangian(debugData);
#      is_mono = (prev_obj>=obj) ? 'O' : 'X';
#      @printf("          L: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
#      prev_obj = obj;
    end

    #
    # update penalties
    mu_t = min(tau*mu_t, mu_max);
    mu_s = min(tau*mu_s, mu_max);
    if DEBUG;
#      debugData["mu_t"] = mu_t;
#      debugData["mu_s"] = mu_s;
#      (obj,rerr,spar) = compute_lagrangian(debugData);
#      is_mono = (prev_obj>=obj) ? 'O' : 'X';
#      @printf("          M: %c obj=%.10f (rerr=%.10f, spar=%.10f)\n", is_mono, obj, rerr, spar);
#      prev_obj = obj;
    end


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


  return (s, z);
end
