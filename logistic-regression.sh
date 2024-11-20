echo -------------- Logistic Regression -------------- > logistic-regression.txt && \
echo Teste 1: >> logistic-regression.txt && \
timeout 120 python3 ./logistic-regression.py 0.1 True >> logistic-regression.txt 2>&1 || echo "Timeout no Teste 1" >> logistic-regression.txt && \
echo ===================================================================== >> logistic-regression.txt && \

echo Teste 2: >> logistic-regression.txt && \
timeout 120 python3 ./logistic-regression.py 0.01 True >> logistic-regression.txt 2>&1 || echo "Timeout no Teste 2" >> logistic-regression.txt && \
echo ===================================================================== >> logistic-regression.txt && \

echo Teste 3: >> logistic-regression.txt && \
timeout 120 python3 ./logistic-regression.py 0.001 True >> logistic-regression.txt 2>&1 || echo "Timeout no Teste 3" >> logistic-regression.txt && \
echo ===================================================================== >> logistic-regression.txt && \

echo Teste 4: >> logistic-regression.txt && \
timeout 120 python3 ./logistic-regression.py 0.0001 True >> logistic-regression.txt 2>&1 || echo "Timeout no Teste 4" >> logistic-regression.txt
