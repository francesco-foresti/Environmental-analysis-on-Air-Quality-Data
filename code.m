clc
clearvars

%% Caricamento del dataset e eliminazione dati non validi %%
% Dopo aver caricato il dataset assegniamo i nomi alle variabili e
% eliminiamo i NaN
T = readtable("Dataset");
T.Properties.VariableNames = {'id', 'Data', 'Precipitazioni', 'Temperatura'...
    'Umidita', 'Radiazione', 'Vento', 'Direzione', 'NO', 'NO2', 'CO', 'O3'};

DATA = rmmissing(T(:,:));

%% QUESITO 1%%

%% 1.1) Introduzione %%
% Plottiamo una matrice di scatterplot tra regressori tramite il comando
% plotmatrix
% [S,AX,BigAx,H,HAx] = plotmatrix(DATA{:, 3:end});
% AX(1,1).YLabel.String='Precipitazioni';
% AX(2,1).YLabel.String='Temperatura';
% AX(3,1).YLabel.String='Umidità';
% AX(4,1).YLabel.String='Radiazione';
% AX(5,1).YLabel.String='Vento';
% AX(6,1).YLabel.String='Direzione';
% AX(7,1).YLabel.String='NO';
% AX(8,1).YLabel.String='NO2';
% AX(9,1).YLabel.String='CO';
% AX(10,1).YLabel.String='O3';
% AX(10,1).XLabel.String='Precipitazioni';
% AX(10,2).XLabel.String='Temperatura';
% AX(10,3).XLabel.String='Umidità';
% AX(10,4).XLabel.String='Radiazione';
% AX(10,5).XLabel.String='Velocità vento';
% AX(10,6).XLabel.String='Direzione vento';
% AX(10,7).XLabel.String='Ossido di azoto';
% AX(10,8).XLabel.String='Biossido di azoto';
% AX(10,9).XLabel.String='Monossido di carbonio';
% AX(10,10).XLabel.String='Ozono';

% Verifichiamo la correlazione vista nei grafici precedenti tramite la matrice di
% correlazione
corr_matrix = corr(DATA{:,3:end});
matrice_rho = array2table(corr_matrix, 'VariableNames' ,{'Precipitazione',...
    'Temperatura','Umidità','Radiazione','Vel Vento','Dir Vento','NO', 'NO2','CO', 'O3' }, ...
    'RowNames',{'Precipitazione',...
    'Temperatura','Umidità','Radiazione','Vel Vento','Dir Vento','NO', 'NO2','CO', 'O3'})

% Lo studio si concentrerà su O3. Vediamo subito che O3 è fortemente
% correlato con Temperatura (rho = 0.83), Umidità (rho = -0.62856) e NO2 (-0.604), ma
% ha anche una buona correlazione con NO e Radiazione (rho = 0.52)
% Tuttavia vediamo che NO e NO2 sono molto correlati, come anche CO con NO.
% Dobbiamo fare quindi attenzione nello scegliere quali di questi
% regressori inserire nel modello così da non incorrere in overfitting.

%% Modello di regressione lineare %%
% Iniziamo l'analisi con dei modelli di regressione lineare semplici tra
% l'O3 e i regressori che abbiamo visto essere più correalti ad esso.

lm_O3_Temp = fitlm(DATA.Temperatura, DATA.O3);
R2_O3_Temp = lm_O3_Temp.Rsquared.Ordinary;

lm_O3_Umidita = fitlm(DATA.Umidita, DATA.O3);
R2_O3_Umidita = lm_O3_Umidita.Rsquared.Ordinary;

lm_O3_NO2 = fitlm(DATA.NO2, DATA.O3);
R2_O3_NO2 = lm_O3_NO2.Rsquared.Ordinary;

% Temperatura e Umidita hanno un R2 più alto rispetto a NO2

%% 1.2) Modello a 2 Regressori: modello di regressione lineare multipla a 2 regressori %%
% Iniziamo con l'analisi di un modello con due regressori, in particolare,
% visti i risultati ottenuti al punto precedente, decidiamo di inserire nel
% modello la Temperatura e l'Umidità.

n = length(DATA.O3);
Y = DATA.O3;
X = [ones(n, 1) DATA.Temperatura DATA.Umidita];

% Verifica che il det(X'X) > 0
det(X'*X);

% Stima di beta hat e y hat
B_hat = (X'*X)\X'*Y;
y_hat = X*B_hat;

% Calcolo di devianza totale, residua, spiegata e di R^2
mY = mean(Y);
Dtot = sum((Y-mY).^2);
Dres = sum((Y-y_hat).^2);
Dsp = sum((y_hat-mY).^2);
R2 = 1-(Dres/Dtot);

% Calcolo s^2
k = 2;
s2e = Dres/(n - k - 1);
s = sqrt(s2e);

% Analisi dei residui
residuals_Temp_Umid = Y - y_hat;

% I residui si distribuiscono attorno allo zero?
% (questo verifica è banale, deriva dal metodo dei minimi quadrati)
mean(residuals_Temp_Umid);

% I residui si distribuiscono come una normale?
% Se così fosse possiamo fare IC e Test di Ipotesi sui Beta.
% histfit(residuals_Temp_Umid);

% Graficamente sembra che i residui si distribuiscano come una normale.
% Eseguiamo un test Jarque-Bera.
alpha = .05;
[h1,p1,jbstat1,critval1] = jbtest(residuals_Temp_Umid, alpha);

% Il test Jarque-Bera rifiuta l'ipotesi nulla che i residui provengano
% da una normale.
% Controlliamo che gli errori siano omoschedastici
figure('Name','Omoschedasticità - Modello a 2 Regressori','NumberTitle','off')
plot(y_hat, residuals_Temp_Umid, 'o')
yline(0,'r','LineWidth',1)
xlabel('Valori fittati')
ylabel('Residui')

%controlliamo se i resiudi sono correlati tra di loro
figure('Name','Autocorrelazione - Modello a 2 Regressori','NumberTitle','off')
autocorr(residuals_Temp_Umid)
xlabel('Lag')
ylabel('Autocorrelazione dei Residui')

%Calcoliamo la matrice δ
delta = (X'*residuals_Temp_Umid)/n;

% Delta ha tutti i componenti ≃ 0 quindi possiamo concludere che la stima
% dei Beta sia non distorta

%% 1.3) Analisi di ‘Aumento numero Regressori’ tramite Crossvalidazione %%
% Vogliamo verificare se abbia senso aggiungere altri regressori al modello di
% regressione lineare analizzato in precedenza. Per fare ciò implementiamo
% un ciclo di cross-validazione dal quale otteniamo degli MSE su modelli
% con numero di regressori diversi e, aiutandoci con l'R2, scegliamo il
% modello più adatto.
ones_v = ones(length(DATA.Temperatura), 1);
x_2 = [DATA.Temperatura DATA.Umidita];
x_3 = [x_2 DATA.NO2];
x_4 = [x_3 DATA.Radiazione];
x_5 = [x_4 DATA.NO];
x_6 = [x_5 DATA.CO];
x_7 = [x_6 DATA.Vento];
x_8 = [x_7 DATA.Direzione];
x_9 = [x_8 DATA.Precipitazioni];

figure('Name','1.3 - Crossvalidazione ','NumberTitle','off')
subplot(2, 1, 1)
title('Grafico EQM')
ylabel('EQM')
xlabel('Numero Regressori')

regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));

n_regressori = [2, 3, 4, 5, 6, 7, 8, 9];
for i = 1:10
    mse2 = crossval('mse',[ones_v x_2],Y,'Predfun',regf, 'kfold', 10);
    mse3 = crossval('mse',[ones_v x_3],Y,'Predfun',regf, 'kfold', 10);
    mse4 = crossval('mse',[ones_v x_4],Y,'Predfun',regf, 'kfold', 10);
    mse5 = crossval('mse',[ones_v x_5],Y,'Predfun',regf, 'kfold', 10);
    mse6 = crossval('mse',[ones_v x_6],Y,'Predfun',regf, 'kfold', 10);
    mse7 = crossval('mse',[ones_v x_7],Y,'Predfun',regf, 'kfold', 10);
    mse8 = crossval('mse',[ones_v x_8],Y,'Predfun',regf, 'kfold', 10);
    mse9 = crossval('mse',[ones_v x_9],Y,'Predfun',regf, 'kfold', 10);
    Vettore_mse = [mse2 mse3 mse4 mse5 mse6 mse7 mse8 mse9];
    
    hold on
    plot(n_regressori, Vettore_mse)
end

% Per semplicità calcoliamo gli R2 dei modelli con il fitlm
r2 = fitlm(x_2, Y).Rsquared.Ordinary;
r3 = fitlm(x_3, Y).Rsquared.Ordinary;
r4 = fitlm(x_4, Y).Rsquared.Ordinary;
r5 = fitlm(x_5, Y).Rsquared.Ordinary;
r6 = fitlm(x_6, Y).Rsquared.Ordinary;
r7 = fitlm(x_7, Y).Rsquared.Ordinary;
r8 = fitlm(x_8, Y).Rsquared.Ordinary;
r9 = fitlm(x_9, Y).Rsquared.Ordinary;
Vettore_R2 = [r2 r3 r4 r5 r6 r7 r8 r9];

subplot(2,1,2)
scatter([2 3 4 5 6 7 8 9],Vettore_R2)
title('Grafico R2')
xlabel('Numero Regressori')
ylabel('Valore R2')

% Analizzando i grafici ottenuti nei passi precedenti, all'aggiunta del 3°
% regressore possiamo apprezzare una drastica diminuzione dell'EQM
% accompagnata poi da una lieve diminuzione di esso con l'aumento del
% numero di regressori. Contemporaneamente anche l'R2 al 3° regressore ha
% un aumento significativo accompagnato poi da un lieve incremento con l'aumentare del
% numero di regressori (d'altronde sappiamo che all'aumentare del numero
% dei regressori, R2 non può far altro che aumentare).
% Entrambe le analisi ci portano alla conclusione di scegliere un modello con 3 regressori
% (evitando così overfitting).
% La scelta tiene conto anche della scarsa correlazione tra O3 e regressori
% come Precipitazioni, Vento e Direzione Vento, e tiene conto del fatto che
% regressori come NO e CO sono correlati con regressori come NO2 che fa
% parte del modello scelto.

%% 1.4) Modello a 3 Regressori: analisi del modello a 3 regressori ottenuto dalla crossvalidazione %%
% Eseguiamo una analisi simile a quella svolta in precedenza, soffermandoci
% a verificare le ipotesi per poter fare inferenza sui Beta_hat.
Y1 = DATA.O3;
X1 = [ones(n, 1) DATA.Temperatura DATA.Umidita DATA.NO2];

% Verifica che il det(X1'X1) > 0
det(X1'*X1);

% Stima di beta hat e y hat
B_hat_1 = (X1'*X1)\X1'*Y1;
y_hat_1 = X1*B_hat_1;

% Calcolo di devianza totale, residua, spiegata e di R^2
mY = mean(Y1);
Dtot_1 = sum((Y1-mY).^2);
Dres_1 = sum((Y1-y_hat_1).^2);
Dsp_1 = sum((y_hat_1-mY).^2);
R2_1 = 1-(Dres_1/Dtot_1);

% Calcolo s^2
k = 3;
s2e_1 = Dres_1/(n - k - 1);
s_1 = sqrt(s2e_1);

% Analisi dei residui
residuals_Temp_Umid_NO2 = Y1 - y_hat_1;

% I residui si distribuiscono attorno allo zero?
% (questo verifica è banale, deriva dal metodo dei minimi quadrati)
mean(residuals_Temp_Umid_NO2);

% Ci interessa la normalità dei Beta_hat per eseguire inferenza.
% Dalla teoria sappiamo che abbiamo i Beta_hat si distribuiscono come una normale
% se:
%   1. i residui sono normali iid con media 0 e varianza SigmaQuadro
% figure('Name','Analisi dei Residui','NumberTitle','off');
% subplot(2,2,1);
% title('Histfit');
% histfit(residuals_Temp_Umid_NO2);

% Graficamente abbiamo dei dubbi che i residui si distribuiscano come una
% normale.
% Eseguiamo un test Jarque-Bera.
alpha = .05;
[h,p,jbstat,critval] = jbtest(residuals_Temp_Umid_NO2, alpha);

% Il test Jarque-Bera rifiuta l'ipotesi nulla che i residui provengano
% da una normale. A questo punto per capire se i Beta_hat si distribuiscono
% come una normale possiamo affidarci alla numerosità dei campioni:
%   2. se gli errori sono iid con media 0 e varianza sigma quadro
%      (omoschedastici) possiamo sfruttare il TLC.
% Verifichiamo quindi se gli errori sono omoschedastici con un plot dei
% residui.
% subplot(2,2,2)
% title('Omoschedasticità')
% plotResiduals(fitlm(DATA(:, 3:end),'O3 ~ Temperatura + Umidita + NO2'),...
%     'fitted')

% vediamo che i punti non sono dispersi allo stesso modo, ma tendono ad avere
% una dispersione maggiore nella parte destra del grafico. Quindi i residui
% non sono omoschedastici e non possiamo fare inferenza sui coefficenti
% beta in modo semplice.
% Giunti a questo punto, per verificare che le nostre stime OLS non siano
% distorte controlliamo la matrice δ
delta_1_4 = (X1'*residuals_Temp_Umid_NO2)/n;

% δ ha tutti i componenti ≃ 0 di conseguenza possiamo concludere che la
% stima dei coefficienti beta sia non distorta asintoticamente.

%% 1.5) Modello Polinomiale: modello di regressione polinomiale %%
% La temperatura ha un ottima correlazione con O3. Dando uno sguardo al
% risultato del plotmatrix inoltre si può vedere come la relazione tra O3 e
% Temperatura sembra avere un andamento non rettilieno. Per questo motivo
% decidiamo di provare a spiegare l'O3 con un modello polinomiale.

n = length(DATA.O3);
Y = DATA.O3;
X_P = [ones(n, 1) DATA.Temperatura DATA.Temperatura.^2];

% Verifica che det(X'X) > 0
det(X_P'*X_P);

% Stima di beta hat e y hat
B_hat_P = (X_P'*X_P)\X_P'*Y;
y_hat_P = X_P*B_hat_P;

% Calcolo di devianza totale, residua, spiegata e di R^2
mY = mean(Y);
Dtot_P = sum((Y-mY).^2);
Dres_P = sum((Y-y_hat_P).^2);
Dsp_P = sum((y_hat_P-mY).^2);
R2_P = 1-(Dres_P/Dtot_P);

% Calcolo s^2
k_P = 2;
s2e_P = Dres_P/(n - k_P - 1);
s_P = sqrt(s2e_P);

% Analisi dei residui
residuals_P = Y - y_hat_P;

% I residui si distribuiscono attorno allo zero?
% (questo verifica è banale, deriva dal metodo dei minimi quadrati)
mean(residuals_P);

% I residui si distribuiscono come una normale?
% Se così fosse possiamo fare IC e Test di Ipotesi sui Beta.
% histfit(residuals_P)

% Graficamente sembra che i residui si distribuiscano come una normale.
% Eseguiamo un test Jarque-Bera.
alpha = .05;
[h1,p1,jbstat1,critval1] = jbtest(residuals_P, alpha);

% Il test Jarque-Bera rifiuta l'ipotesi nulla che i residui provengano
% da una normale.
% Controlliamo che gli errori siano omoschedastici
% figure
% plot(y_hat_P, residuals_P, 'o')

%Calcoliamo la matrice δ
delta_P = (X_P'*residuals_P)/n;

% cross-validazione per scegliere il grado del polinomio
ones_v = ones(length(DATA.Temperatura), 1);
xp_1 = [DATA.Temperatura];
xp_2 = [xp_1 DATA.Temperatura.^2];
xp_3 = [xp_2 DATA.Temperatura.^3];
xp_4 = [xp_3 DATA.Temperatura.^4];
xp_5 = [xp_4 DATA.Temperatura.^5];
xp_6 = [xp_5 DATA.Temperatura.^6];
xp_7 = [xp_6 DATA.Temperatura.^7];

% figure('Name','Cross-validazione regressione polinomiale','NumberTitle','off')
% subplot(2, 1, 1)
% title('Grafico EQM')
% ylabel('EQM')
% xlabel('Grado del polinomio')

regf=@(XTRAIN,yhattrain,XTEST)(XTEST*regress(yhattrain,XTRAIN));

n_regressori = [1 2, 3, 4, 5, 6, 7];
for i = 1:10
    mse1p = crossval('mse',[ones_v xp_1],Y,'Predfun',regf, 'kfold', 10);
    mse2p = crossval('mse',[ones_v xp_2],Y,'Predfun',regf, 'kfold', 10);
    mse3p = crossval('mse',[ones_v xp_3],Y,'Predfun',regf, 'kfold', 10);
    mse4p = crossval('mse',[ones_v xp_4],Y,'Predfun',regf, 'kfold', 10);
    mse5p = crossval('mse',[ones_v xp_5],Y,'Predfun',regf, 'kfold', 10);
    mse6p = crossval('mse',[ones_v xp_6],Y,'Predfun',regf, 'kfold', 10);
    mse7p = crossval('mse',[ones_v xp_7],Y,'Predfun',regf, 'kfold', 10);
    Vettore_mse_p = [mse1p mse2p mse3p mse4p mse5p mse6p mse7p];
    
    %     hold on
    %     plot(n_regressori, Vettore_mse_p)
end

% Per semplicità calcoliamo gli R2 dei modelli con il fitlm
r1p = fitlm(xp_1, Y).Rsquared.Ordinary;
r2p = fitlm(xp_2, Y).Rsquared.Ordinary;
r3p = fitlm(xp_3, Y).Rsquared.Ordinary;
r4p = fitlm(xp_4, Y).Rsquared.Ordinary;
r5p = fitlm(xp_5, Y).Rsquared.Ordinary;
r6p = fitlm(xp_6, Y).Rsquared.Ordinary;
r7p = fitlm(xp_7, Y).Rsquared.Ordinary;
Vettore_R2_p = [r1p r2p r3p r4p r5p r6p r7p];

% subplot(2,1,2)
% scatter([1 2 3 4 5 6 7],Vettore_R2_p)
% title('Grafico R2')
% xlabel('Grado del polinomio')
% ylabel('Valore R2')

% dal grafico dell'MSE possiamo notare come oltre il secondo grado
% l'MSE non diminuisce in modo significativo.
% Dal grafico dell'R2 invece vediamo come oltre il secondo grado
% l'R2 non aumenti. Quindi li grado del polinomio da scegliere è due, che è
% quello che abbiamo anlizzato in precedenza.

%% 1.6) Conclusioni Quesito 1 %%

m2 = [s2e R2];
m3 = [s2e_1 R2_1];
mp = [s2e_P R2_P];
matriceConfronto = array2table([m2; m3; mp], 'VariableNames', ...
    {'MSE', 'R2'}, 'RowNames', {'Modello a 2 Regressori', ...
    'Modello a 3 Regressori', 'Modello Polinomiale'})

%% QUESITO 2 %%

%% 2.1) Confronto B-Splines con Basi di Fourier tramite Crossvalidazione %%
% Vogliamo modellizzare l'andamento della Temperatura nel tempo. Dal
% momento che la temperatura è periodica decidiamo di usare le basi di
% fourier.

X_B = 1:1:length(DATA.Temperatura);

% determiniamo il numero di basi tramite gcv
iter = 1;
num_iteration = 100;
for i=1:num_iteration
    %disp(['Crossval iteration:',num2str(iter), ' di ', num2str(num_iteration)])
    l = (max(X_B)-min(X_B))/i;
    knots = min(X_B):l:max(X_B);
    nknots = length(knots); % numero di nodi
    norder = 4; % ordine
    interior_knots = nknots - 2; % nodi interni
    nbasis = norder + interior_knots; % numero di basi
    rangevalue = [min(X_B), max(X_B)]; % intervallo di valutazione
    save_basi(iter) = nbasis;
    % Fourier basis
    basis1 = create_fourier_basis(rangevalue,nbasis);
    basismat1 = eval_basis(X_B, basis1);
    Cmap1 = (basismat1' * basismat1) \ (basismat1');
    chat1 = Cmap1 * DATA.Temperatura;
    yhat1 = basismat1 * chat1;
    
    gcv(1,iter) = sum(((DATA.Temperatura - yhat1)./(1 - nbasis/n)).^2)/n;
    
    if nbasis == 50
        figure('Name','Basi di Fourier 50 basi','NumberTitle','off')
        scatter(X_B, DATA.Temperatura);
        hold on
        plot(X_B, yhat, 'g')
        title('Basi di Fourier 50 basi')
        xlabel('Osservazioni')
        ylabel('Valori Temperatura')
        legend('Valori Osservati', 'Funzione stimata')
        hold off
        
        % calcoliamo RSS e sigma2 hat
        RSS = sum((DATA.Temperatura - yhat1).^2);
        sigma_hat = RSS / (n - nbasis);
        
        alpha = .05;
        z_alpha = norminv(1 - (alpha)/2);
        
%--**    IL SEGUENTE CODICE È COMPUTAZIONALMENTE MOLTO IMPEGNATIVO **--%
%        il codice è in grado di calcolare gli IC per i singoli elementi 
%        del vettore yhat1.
%
%        Smat = basismat1*Cmap1;
%        df = trace(Smat);
%        df = nbasis
%        sigma_square_hat = RSS / (length(DATA.Temperatura) - df)
%        var_cov_yhat = sigma_square_hat*(Smat'*Smat)
%        varYhat = diag(var_cov_yhat)       
%        Lower = yhat1 - z_alpha*sqrt(varYhat);
%        Upper = yhat1 + z_alpha*sqrt(varYhat);
%        figure
%        hold on
%        scatter(X_B,  DATA.Temperatura, 5,"cyan");
%        hold on
%        plot(X_B, yhat1, "black")
%        hold on
%        plot(X_B, Lower, ':r', 'LineWidth', 1)
%        hold on
%        plot(X_B, Upper, ':r', 'LineWidth', 1)
%        legend('Dati Osservati', 'Funzione Stimata', 'Upper IC', 'Lower IC')
%        hold off
    end
    
    % B Spline
    basis = create_bspline_basis(rangevalue, nbasis, norder, knots);
    basismat = eval_basis(X_B, basis);
    Cmap = (basismat' * basismat) \ (basismat');
    chat = Cmap * DATA.Temperatura;
    yhat = basismat * chat;
    
    % questo pezzo di codice serve per plottare una bspline di ordine 4 con
    % 50 basi per effettuare il confronto con le basi di fourier. Tra i due
    % grafici tuttavia non ci sono grosse differenze, sono entrambi molto
    % smooth e seguono bene la funzione
    %     if nbasis == 50
    %         figure('Name','BSpline di ordine 4 con 50 basi','NumberTitle','off')
    %         scatter(X_B, DATA.Temperatura);
    %         hold on
    %         plot(X_B, yhat, 'g')
    %         title('BSpline di ordine 4 con 50 basi')
    %         xlabel('Osservazioni')
    %         ylabel('Valori Temperatura')
    %         legend('Valori Osservati', 'Funzione stimata')
    %         hold off
    %     end
    
    gcv(2,iter) = sum(((DATA.Temperatura - yhat)./(1 - nbasis/n)).^2)/n;
    iter = iter + 1;
end

% eseguiamo il codice sopra aumentando il numero di nodi.
% questo step lo eseguiamo manualmente.
% Proviamo con 500, 1000, 1500, 2000
l = (max(X_B)-min(X_B))/1000;
knots = min(X_B):l:max(X_B);
nknots = length(knots); % numero di nodi
norder = 4; % ordine
interior_knots = nknots - 2; % nodi interni
nbasis = norder + interior_knots; % numero di basi

save_basi(iter) = nbasis;
rangevalue = [min(X_B), max(X_B)]; % intervallo di valutazione

% Fourier basis
basis1 = create_fourier_basis(rangevalue,nbasis);
basismat1 = eval_basis(X_B, basis1);
Cmap1 = (basismat1' * basismat1) \ (basismat1');
chat1 = Cmap1 * DATA.Temperatura;
yhat1 = basismat1 * chat1;

gcv(1,iter) = sum(((DATA.Temperatura - yhat1)./(1 - nbasis/n)).^2)/n;

% B Spline
basis = create_bspline_basis(rangevalue, nbasis, norder, knots);
basismat = eval_basis(X_B, basis);
Cmap = (basismat' * basismat) \ (basismat');
chat = Cmap * DATA.Temperatura;
yhat = basismat * chat;

gcv(2,iter) = sum(((DATA.Temperatura - yhat)./(1 - nbasis/n)).^2)/n;
iter = iter + 1;

% Plottiamo l'andamento dell'MSE rispetto al numero di basi. Possiamo
% ossevare come l'MSE tenda a diminuire mano a mano che il numero di basi
% aumenta e che tende ad andare a 0. 
% Nel grafico al momento mostriamo solo i valori calcolati fino a
% num_iteration. Se si toglie questo vincolo si possono vedere anche quello
% generati dal codice sopra con un numero di basi a scelta.
figure('Name','Andamento MSE rispetto al numero di basi','NumberTitle','off')
plot(save_basi(1, 1:num_iteration), gcv(1, 1:num_iteration), 'g')
hold on
plot(save_basi(1, 1:num_iteration), gcv(2, 1:num_iteration), 'r')
xlabel('Numero di Basi')
ylabel('GCV')
legend('Fourier','B-Spline di ordine 4')

% Tuttavia se guardiamo i plot dei valori stimati con numero di basi alto
% (già con un numero di basi sopra il 100), osserviamo che c'è overfitting.
% Invece se prendiamo un numero di basi inferiore, come 50, vediamo che il
% grafico è molto più liscio e segue meglio l'andamento che immaginiamo
% possa avere la temperatua. Quindi scegliamo di usare 50 basi.
figure
title('Fourier Basis vs Real Points')
hold on
scatter(X_B, DATA.Temperatura)
plot(X_B, yhat1)
ylabel('Temperatura')
xlabel('Numero osservazione')
legend('Osservazioni','Fourier 1000 basi')