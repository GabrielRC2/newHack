"""Contratos HTTP e valores controlados da API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PapelCamera(str, Enum):
    ENTRADA = "ENTRADA"
    SAIDA = "SAIDA"


class DecisaoReconhecimento(str, Enum):
    ENTRADA_IDENTIFICADA = "ENTRADA_IDENTIFICADA"
    CADASTRO_NECESSARIO = "CADASTRO_NECESSARIO"
    ROSTO_DESCONHECIDO = "ROSTO_DESCONHECIDO"
    IDENTIFICADO_SEM_AULA_ATIVA = "IDENTIFICADO_SEM_AULA_ATIVA"
    IDENTIFICADO_ANTES_DO_CORTE = "IDENTIFICADO_ANTES_DO_CORTE"
    PRESENCA_REGISTRADA = "PRESENCA_REGISTRADA"
    PRESENCA_JA_REGISTRADA = "PRESENCA_JA_REGISTRADA"


class AlunoCreate(BaseModel):
    nome: str = Field(min_length=1, max_length=120)
    matricula: str = Field(min_length=1, max_length=50)

    @field_validator("nome", "matricula")
    @classmethod
    def remove_blank_space(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("não pode ser vazio")
        return cleaned_value


class AlunoResponse(AlunoCreate):
    id: int
    criado_em: datetime


class CameraCreate(BaseModel):
    codigo: str = Field(min_length=1, max_length=50)
    nome: str = Field(min_length=1, max_length=120)
    papel: PapelCamera

    @field_validator("codigo", "nome")
    @classmethod
    def remove_blank_space(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("não pode ser vazio")
        return cleaned_value


class CameraResponse(CameraCreate):
    id: int
    criada_em: datetime


class SessaoAulaCreate(BaseModel):
    nome: str = Field(min_length=1, max_length=120)
    camera_saida_codigo: str = Field(min_length=1, max_length=50)
    inicio_em: datetime
    fim_em: datetime

    @field_validator("nome", "camera_saida_codigo")
    @classmethod
    def remove_blank_space(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("não pode ser vazio")
        return cleaned_value

    @model_validator(mode="after")
    def validate_time_range(self) -> "SessaoAulaCreate":
        ensure_timezone("inicio_em", self.inicio_em)
        ensure_timezone("fim_em", self.fim_em)
        if self.fim_em <= self.inicio_em:
            raise ValueError("fim_em deve ser posterior a inicio_em")
        return self


class SessaoAulaResponse(BaseModel):
    id: int
    nome: str
    camera_saida_codigo: str
    inicio_em: datetime
    fim_em: datetime
    corte_em: datetime
    criado_em: datetime


class EventoReconhecimentoCreate(BaseModel):
    """Entrada temporária para o MVP, antes da integração de embeddings."""

    camera_codigo: str = Field(min_length=1, max_length=50)
    aluno_id: Optional[int] = Field(default=None, ge=1)
    ocorreu_em: datetime

    @field_validator("camera_codigo")
    @classmethod
    def remove_blank_space(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("não pode ser vazio")
        return cleaned_value

    @field_validator("ocorreu_em")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        ensure_timezone("ocorreu_em", value)
        return value


class EventoReconhecimentoResponse(BaseModel):
    id: int
    camera_codigo: str
    aluno_id: Optional[int]
    decisao: DecisaoReconhecimento
    sessao_aula_id: Optional[int]
    ocorreu_em: datetime
    corte_em: Optional[datetime]


class PresencaResponse(BaseModel):
    aluno_id: int
    aluno_nome: str
    matricula: str
    evento_id: int
    registrado_em: datetime


def ensure_timezone(field_name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} deve informar o fuso horário")


def as_utc(value: datetime) -> datetime:
    """Normaliza datas de API para UTC antes de persistir ou comparar."""

    return value.astimezone(timezone.utc)
