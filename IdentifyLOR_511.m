function Ind_coin = IdentifyLOR_511(energy, CorrectedTime, CoincidenceTime)

Ind_511 = find(abs(energy-0.511)<0.0001);
indmap = 1:numel(CorrectedTime);
indmap = indmap(Ind_511);
CorrectedTime_511 = CorrectedTime(Ind_511);

[sortedtime_511, sortInd_511] = sort(CorrectedTime_511);


difftime = diff(sortedtime_511);
difftime2 = difftime(1:end-1) + difftime(2:end);
newsortInd_coin = find(difftime<CoincidenceTime);
newsortInd_coin2 = find(difftime2<CoincidenceTime);
newsortInd_multiplecoin = union(union(newsortInd_coin2,newsortInd_coin2+1),newsortInd_coin2+2);
newsortInd_coin_cleaned = setdiff(newsortInd_coin,newsortInd_multiplecoin);

sortInd_coin1 = sortInd_511(newsortInd_coin_cleaned);
sortInd_coin2 = sortInd_511(newsortInd_coin_cleaned+1);

Ind_coin1 = indmap(sortInd_coin1);
Ind_coin2 = indmap(sortInd_coin2);

Ind_coin = [Ind_coin1(:) Ind_coin2(:)];

% test2 = [Mega(Ind_coin1,:) Mega(Ind_coin2,:) Mega(Ind_coin2,1)-Mega(Ind_coin1,1)];
% test3 = [Mega(indmap(sortInd_511),:)];

