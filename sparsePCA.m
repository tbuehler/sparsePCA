% Computes the sparse PCA components using the nonlinear inverse power
% method (NIPM), as described in the paper
%
% M. Hein and T. Buehler
% An Inverse Power Method for Nonlinear Eigenproblems with Applications 
% in 1-Spectral Clustering and Sparse PCA
% In Advances in Neural Information Processing Systems 23 (NIPS 2010)
% Available online at http://arxiv.org/abs/1012.0774
%
% Usage:
% [cards, vars, Z] = sparsePCA(X, card_min);
% [cards, vars, Z] = sparsePCA(X, card_min, card_max);
% [cards, vars, Z] = sparsePCA(X, card_min, card_max, num_runs);
% [cards, vars, Z] = sparsePCA(X, card_min, card_max, num_runs, verbosity);
%
% X         data matrix (num x dim)
% card_min  desired number of non-sparse components of output (cardinality)
% card_max  if specified, all vectors with cardinality values in intervall 
%           [card_min,card_max] are computed (default: card_max=card_min)
% num_runs  number of runs of inverse power method with random 
%           initialization (default: 10)
% verbosity determines frequency of console output (values 0-2, default: 1)
%
% cards     the cardinalities (number of nonzero components) of the 
%           returned vectors 
% vars      the corresponding variances
% Z         the sparse principal components
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Copyright 2010-15 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de
function [cards, vars, Z]= sparsePCA(X, card_min, card_max, num_runs, verbosity)

    if (nargin<5), verbosity=1; end
    if (nargin<4), num_runs=10; end
    if (nargin<3), card_max=card_min; end

    assert(card_min>0,'Wrong usage. Cardinality has to positive.');
    assert(card_max>=card_min,'Wrong usage. card_max cannot be smaller than card_min.');
    assert(num_runs>=0,'Wrong usage. num_runs cannot be negative.');
    assert(card_max<=size(X,2),'Wrong usage. Cardinality can not be larger than dim.');

    [num,dim]=size(X);
    gam_left=0;
    gam_right=1;

    % compute startvector
    norm_a_i=zeros(dim,1);
    for i=1:dim, norm_a_i(i)=norm(X(:,i)); end
    [rho_max,i_max]=max(norm_a_i);
    start1=zeros(dim,1);
    start1(i_max)=1;

    % output
    cards=zeros(card_max-card_min+1,1);
    gammas=zeros(card_max-card_min+1,1);
    vars=zeros(card_max-card_min+1,1);
    Z=zeros(dim,card_max-card_min+1);
    lambdas=zeros(card_max-card_min+1,1);
    found_cards=zeros(card_max-card_min+1,1);
    
    card0=round((card_max+card_min)/2);

    % searches for a vector with cardinality card0 via binary search (and
    % stores the solutions for other cardinalities it finds along the way)
    [cards, vars, Z, gammas, lambdas, found_cards] = binSearch(X, card0, gam_left, gam_right, ... 
        num_runs, start1, cards, vars, Z, gammas, lambdas, found_cards, card_min, card_max, verbosity);

    % repeat this until all gaps are filled
    ix1=find(found_cards==0,1);
    while(~isempty(ix1))
        card_min_temp=card_min-1+ix1;
        if(card_min_temp>card_min)
            gam_right=gammas(card_min_temp-card_min);
        else 
            gam_right= 1;
        end

        ix2=find(found_cards(ix1:end)>0,1);
        if(~isempty(ix2))
            if found_cards(ix2)<inf
                card_max_temp=card_min-1+ix1-1+ix2-1;
                gam_left=gammas(card_max_temp-card_min+2);
            else
                card_max_temp=card_min-1+ix1-1+ix2-1;
                gam_left=0;
            end
        else
            card_max_temp=card_max;
            gam_left=0;
        end
        
        card0=round((card_max_temp+card_min_temp)/2);

        if (verbosity>1)
            fprintf('card_min_temp= %d card_max_temp= %d gam_left=%.5g gam_right=%.5g\n',card_min_temp,card_max_temp,gam_left,gam_right);
        end

        [cards, vars, Z, gammas, lambdas, found_cards] = binSearch(X, card0, ... 
            gam_left, gam_right, num_runs, start1, cards, vars, Z, gammas, ... 
            lambdas, found_cards, card_min, card_max, verbosity);

        ix1=find(found_cards==0,1);
    end

    isChanging=true;
    vars_old=vars;
    while(isChanging)
        % check if variance is monotonically increasing
        cur_var=vars(1);
        curz=Z(:,1);
        for k=2:length(vars)
            new_var=vars(k);
            if(new_var<cur_var || cards(k)==inf)
                % take sparsity pattern from previous one + add one nonzero
                % component -> increasse in variance guaranteed
                norm_a_i=zeros(dim,1);
                pattern =  abs(curz)>0;
                ind=find(pattern==0);
                temp=X*curz;temp=temp/norm(temp);
                for i=1:length(ind),
                    norm_a_i(ind(i))=(X(:,ind(i))'*temp)^2;
                end
                [rho_max,i_max]=max(norm_a_i);
                pattern(i_max)=1;
                [z, new_var2]=optimizeVariance(X,pattern);

                vars(k)=new_var2;
                Z(:,k)=z;
                cards(k)=sum(pattern);

                % check if we find something better via invpow
                card0=card_min-1+k;
                gam_left=0;
                gam_right=1;
                [cards, vars, Z, gammas, lambdas, found_cards] = binSearch(X, card0, gam_left, gam_right, ... 
                    num_runs, start1, cards, vars, Z, gammas, lambdas, found_cards, card_min, card_max, verbosity);
            end
            cur_var=vars(k);
            curz=Z(:,k);
        end

        if  norm(vars-vars_old,inf)/norm(vars(vars<inf),inf)< 1E-15
            isChanging=false;
        end
        vars_old=vars;
        if verbosity>1
            fprintf('isChanging=%d  normdiff=%.15g\n',isChanging,norm(vars-vars_old,inf));
        end
    end

    if verbosity>1
        isDecreasing=false;
        cur_var=vars(1);
        for k=2:length(vars)
            new_var=vars(k);
            if new_var< cur_var 
                isDecreasing=true;
            end
            cur_var=new_var;
        end
        fprintf('isChanging=%d isDecreasing=%d \n',isChanging,isDecreasing);
    end	
end


% perform sparse PCA with current value of gamma
function [z, var_new, card_new, lambda_new] = performOneRun(X, gam, maxit, start)
    [z,lambda_new] = invPow(X,gam,maxit,start);
    ind_z=abs(z)>0;
    card_new = sum(ind_z);
    [z,var_new] = optimizeVariance(X,ind_z);
end


% perform sparse PCA with current value of gamma - one run initialized with
% start1 and num_runs random initalizations
function [Z, vars, cards, lambdas, best_card] = performMultipleRuns(X, gam, maxit, start1, num_runs)
    [num,dim]=size(X);
    [Z, vars, cards, lambdas] = performOneRun(X,gam,maxit,start1);

    % keep track of the cardinality corresponding to smallest lambda
    best_lambda = lambdas;
    best_card = cards;

    for l=1:num_runs
        start = randn(dim,1);
        [z, var_new, card_new, lambda_new] = performOneRun(X,gam,maxit,start);

        %find the best variance
        already_seen = false;
        for i = 1:length(cards)
            if cards(i) == card_new
                already_seen = true;
                if var_new > vars(i)
                    vars(i) = var_new;
                    Z(:,i) = z;
                    lambdas(i) = lambda_new;
                end
            end
        end

        % append if cardinality has not been seen before
        if ~already_seen
            Z = [Z z];
            vars = [vars var_new];
            cards = [cards card_new];
            lambdas = [lambdas lambda_new];
        end

        % keep track of the cardinality corresponding to smallest lambda
        if lambda_new < best_lambda
            best_lambda = lambda_new;
            best_card = card_new;
        end
    end
end


% searches for a vector with cardinality card0 via binary search (and
% stores the solutions for other cardinalities it finds along the way)
function [cards, vars, Z, gammas, lambdas, found_cards] = binSearch(X, card0, gam_left, gam_right, num_runs, ... 
    start1, cards, vars, Z, gammas, lambdas, found_cards, card_min_global, card_max_global, verbosity)

    maxit=100;
    epsilon=1E-6;
    isFound=false;
    splitpoint=0.5;
    gam=splitpoint*gam_left+(1-splitpoint)*gam_right;

    while (~isFound && gam_right-gam_left>epsilon)
        [Z_tmp, vars_tmp, cards_tmp, lambdas_tmp, best_card] = performMultipleRuns(X,gam, maxit,start1,num_runs);

        % consider the cardinality corresponding to smallest lambda
        if card0 == best_card
            isFound = true;
        else
            % update gamma boundaries
            if (best_card > card0)
                gam_left = gam;
            elseif (best_card < card0)
                gam_right = gam;
            end
        end

        % discard all values which are outside the cardinality range
        ind = find(cards_tmp >= card_min_global & cards_tmp <= card_max_global);

        if(verbosity>1)
            ind2 =find(cards_tmp < card_min_global | cards_tmp > card_max_global);
            if(~isempty(ind2))
                for l=1:length(ind2)
                    fprintf('Skipping solution with cardinality %d\n',cards_tmp(ind2(l)));
                end
            end
        end

        cards_tmp = cards_tmp(ind);
        vars_tmp = vars_tmp(ind);
        Z_tmp = Z_tmp(:,ind);
        lambdas_tmp = lambdas_tmp(ind);

        % store best results
        for l=1:length(cards_tmp)
            cur_card = cards_tmp(l);
            cur_var = vars_tmp(l);

            if cur_card == best_card
                found_cards(best_card-card_min_global+1) = best_card;
            end

            if vars(cur_card-card_min_global+1) < cur_var

                if verbosity>1
                    if (cards(cur_card-card_min_global+1)==0)
                        fprintf('Found solution with cardinality %d\n',cur_card);
                    else
                        fprintf('Improved solution with cardinality %d\n',cur_card);
                    end
                end
                cards(cur_card-card_min_global+1) = cur_card;
                vars(cur_card-card_min_global+1) = cur_var;
                Z(:,cur_card-card_min_global+1) = Z_tmp(:,l);
                gammas(cur_card-card_min_global+1,:) = gam;
                lambdas(cur_card-card_min_global+1) = lambdas_tmp(l);
            end
        end
        gam=splitpoint*gam_left+(1-splitpoint)*gam_right;

        if(verbosity>1)
            fprintf('gam_left= %.3g gam=%.3g gam_right= %.3g mincard=%d maxcard=%d best_card=%d card0=%d\n', ...
            gam_left,gam,gam_right,min(cards_tmp), max(cards_tmp), best_card,card0);
        end
    end

    % if no vector with cardinality card0 could be found, set entry to inf
    if(~isFound)
        % (might have been found as suboptimal solution)
        if cards(card0-card_min_global+1)==0
            cards(card0-card_min_global+1)=Inf;
            vars(card0-card_min_global+1)=Inf;
        end
        if verbosity>1
            fprintf('Skipping solution with cardinality %d\n',card0);
        end
        found_cards(card0-card_min_global+1)=inf;
    end
end
