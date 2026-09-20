from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StudentCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    enrollment_number: str = Field(min_length=1, max_length=80)
    embedding: list[float] = Field(min_length=2)


class EmbeddingCreate(BaseModel):
    vector: list[float] = Field(min_length=2)


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    enrollment_number: str
    created_at: datetime
    em_aula: bool
    tempo_em_aula: int


class ClassSessionCreate(BaseModel):
    label: str = Field(min_length=1, max_length=160)
    starts_at: datetime
    ends_at: datetime
    minimum_percentage: float = Field(default=0.75, gt=0, le=1)

    @model_validator(mode="after")
    def validate_period(self):
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at deve ser posterior a starts_at")
        return self


class ScheduledSessionCreate(BaseModel):
    class_date: date
    schedule_index: int = Field(ge=1, le=6)
    label: str | None = None
    minimum_percentage: float = Field(default=0.75, gt=0, le=1)


class ClassSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    label: str
    starts_at: datetime
    ends_at: datetime
    minimum_percentage: float
    status: str
    closed_at: datetime | None


class RecognitionRequest(BaseModel):
    # class_id e o id de ClassSession. Mantido assim para o contrato com o modulo de visao.
    class_id: int
    camera_id: str = Field(min_length=1, max_length=80)
    embedding: list[float] = Field(min_length=2)
    timestamp: datetime


class RecognitionOut(BaseModel):
    event_id: int
    status: str
    action: str | None = None
    student_id: int | None = None
    match_score: float | None = None
    message: str


class IntervalOut(BaseModel):
    entered_at: datetime
    exited_at: datetime | None


class AttendanceOut(BaseModel):
    student_id: int
    student_name: str
    enrollment_number: str
    total_seconds: int
    total_minutes: float
    required_seconds: int
    status: str
    em_aula: bool
    intervals: list[IntervalOut]


class FinalizeOut(BaseModel):
    class_id: int
    finalized_students: int
    present: int
    absent: int
