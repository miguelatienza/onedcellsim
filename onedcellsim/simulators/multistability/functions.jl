using LambertW

function velocities(aoverN, zetaf, zetab, Ve_0, kf, kb, k_minus, E, Lf, Lb, L0)
    ##function to update the velocities according to model equations
    local Cf, Cb, betaf, betab, vf, vb, vrf, vrb
    ##Return vf, vb, vrf, vrb
    aoverN=0
    if aoverN==0
        # This is the model which ignores the  force dependence of actin polymerization
        Ve_0 = Ve_0-k_minus
        vrf = (E*(Lf-L0) + zetaf*Ve_0)/(kf+zetaf)
        vf = Ve_0 - vrf

        vrb = (E*(Lb-L0) + zetab*Ve_0)/(kb+zetab)
        vb = vrb -Ve_0
        
    else
        #Use the full model with force dependence of actin polymerization
        Cf=Ve_0*kf*zetaf/(zetaf+kf)
        Cb=Ve_0*kb*zetab/(zetab+kb)

        betaf=kf*(k_minus*zetaf - E*(Lf-L0))/(zetaf+kf)
        betab=kb*(k_minus*zetab - E*(Lb-L0))/(zetab+kb)

        #Cf=1; Cb=1; betaf=1; betab=1; vf=1 ; vb=1; vrf=1; vrb=1
        vf = (1/aoverN*zetaf)* lambertw(aoverN*Cf*exp(aoverN*betaf)) - (E*(Lf-L0) + k_minus*kf)/(zetaf+kf)
        vb = -(1/aoverN*zetab)* lambertw(aoverN*Cb*exp(aoverN*betab)) + (E*(Lb-L0) + k_minus*kb)/(zetab+kb)

        vrf = lambertw(Cf*exp(betaf)) /kf - betaf/kf
        vrb = lambertw(Cb*exp(betab)) /kb - betab/kb
    end
    
    return vf, vb, vrf, vrb
end


function get_vc(E, Lf, Lb, zetac)

    return E*(Lf-Lb)/zetac
end

function etaf(alpha, c1, c2, c3, k_lim, k, vr)
    #function to calculate the noise amplitude
    local toroot, std

    toroot = ((c1*(k_lim-k)) + (c2*exp(abs(vr)/c3)*k))/(alpha*k_lim)            
    std = sqrt(clamp(toroot, 0, Inf))
    
    return std
end

function dk_dt(c1, c2, c3, k_lim, k, k0, vr, alpha, epsilon)
    #function to calculate the derivate of kappa
    local dkdt::Float64, noise::Float64
    
    dkdt = c1*(k_lim-(k-k0)) - c2*exp(abs(vr)/c3)*(k-k0)
    
    noise = epsilon*randn()*etaf(alpha, c1, c2, c3, k_lim, k, vr)
    
    return dkdt + noise
end

function solve_x(t_step, xf, xb, xc, vf, vb, vc, epsilon_l)
    
    local x_step_f, x_step_b, x_step_c
    #noise=t_step*epsilon_l*0
    #print(vf, " ", t_step, " ", randn(), "\n")
    
    # x_step_f = clamp((t_step*vf)+(randn()*noise), (xc-xf)*0.45, Inf)
    # x_step_b = clamp((t_step*vb)+(randn()*noise), -Inf, (xc-xb)*0.45)
    # x_step_c = clamp((t_step*vc)+(randn()*noise), (xb-xc)*0.45, (xf-xc)*0.45)
    x_step_f = t_step*vf
    x_step_b = t_step*vb
    x_step_c = t_step*vc

   
    # x_step_b = t_step*vb*(1+ clamp(randn()*3, -Inf, xc-xb*0.5))
    # x_step_c = t_step*vc*(1+ clamp(randn()*3, xb-xc*0.5, xf-xc*0.5))

    xf+=x_step_f

    xb+=x_step_b

    xc+=x_step_c
    
    return xf, xb, xc
end

function zeta_edge(zeta0, zeta_max, B, Kzeta, nzeta)
    return zeta0 + (zeta_max*B^nzeta)/(Kzeta^nzeta + B^nzeta)
end

function zeta_nuc(b, zeta0, zeta_max, B, Kzeta, nzeta)

    return b*(zeta0 + (zeta_max*B^nzeta)/(Kzeta^nzeta + B^nzeta))
end

function k_lim_func(k0, k_max, B, Kk, nk)
    return k0 + (k_max * B^nk/(Kk^nk + B^nk))
end