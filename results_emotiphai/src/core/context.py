from src.core.db import SessionLocal, scoped_session
from src.services.acquisition import AcquisitionService
from src.kubesense.controller import Controller as KubesenseService
from src.services.movie import MovieService
from src.services.annotation import AnnotationService
from src.services.analysis import AnalysisService
from src.core.config import config


class Context:
    def __init__(self):
        self._db: scoped_session = SessionLocal()
        self._acquisition_service: AcquisitionService = None
        self._kubesense_service: KubesenseService = None
        self._movie_service: MovieService = None
        self._annotation_service: AnnotationService = None
        self._analysis_service: AnalysisService = None

    def close(self):
        self._db.close()

    def commit(self):
        self._db.commit()

    @property
    def kubesense_service(self) -> KubesenseService:
        if self._kubesense_service is None:
            self._kubesense_service = KubesenseService()
            self._kubesense_service.config(config.PORTS, config.CHANNELS, config.DATABASE_URL)
        return self._kubesense_service

    @property
    def acquisition_service(self) -> AcquisitionService:
        if self._acquisition_service is None:
            self._acquisition_service = AcquisitionService(self._db, self.kubesense_service)
        return self._acquisition_service
    
    @property
    def movie_service(self) -> MovieService:
        if self._movie_service is None:
            self._movie_service = MovieService()
        return self._movie_service
    
    @property
    def annotation_service(self) -> AnnotationService:
        if self._annotation_service is None:
            self._annotation_service = AnnotationService(self._db)
        return self._annotation_service

    @property
    def analysis_service(self) -> AnalysisService:
        if self._analysis_service is None:
            self._analysis_service = AnalysisService(self._db)
        return self._analysis_service