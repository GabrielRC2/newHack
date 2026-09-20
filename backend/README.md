# Backend — presença por reconhecimento facial

Este backend trata cada câmera como uma produtora de eventos. Ele não rastreia
trajetória, direção ou linha virtual: a decisão de presença depende somente da
identificação na câmera de saída e do horário do evento.

## Marco 1: regra temporal pronta

Esta primeira entrega contém:

- cadastro de alunos;
- cadastro de câmeras com papel fixo `ENTRADA` ou `SAIDA`;
- sessões de aula vinculadas à câmera de saída;
- eventos de reconhecimento simulados por `aluno_id`;
- presença idempotente: uma presença por aluno e sessão;
- persistência local em SQLite;
- testes das regras principais.

Ainda não há captura de vídeo, detector facial ou embeddings. Esses componentes
entrarão como um adaptador na próxima etapa, sem mudar a regra de presença.

## Regra de presença

Para uma aula de 14:00 a 16:00, o corte é 15:00.

- 14:20 e 14:50: identificado, sem presença;
- exatamente 15:00: identificado, sem presença;
- 15:01 em diante: presença registrada.

O período de uma aula é `[início, fim)`: às 16:00 exatas já não há aula ativa.
Todos os horários são normalizados e armazenados em UTC, embora a API aceite
datas ISO-8601 com qualquer offset explícito.

## Executar

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --reload
```

Abra `http://127.0.0.1:8000/docs` para testar a API interativamente.

## Testar

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Fluxo atual de demonstração

1. `POST /alunos`
2. `POST /cameras` para uma `ENTRADA` e uma `SAIDA`
3. `POST /sessoes-aula` com a câmera de saída e os horários
4. `POST /eventos-reconhecimento` com `camera_codigo`, `aluno_id` e `ocorreu_em`
5. `GET /sessoes-aula/{id}/presencas`

O `aluno_id` no passo 4 é temporário. Na próxima etapa, ele será resolvido por
um serviço de reconhecimento facial a partir de um embedding enviado pela câmera.
