from typing import List, Optional, Any
from dataclasses import dataclass


@dataclass
class Category:
    _id: str
    name: str

    @staticmethod
    def from_dict(obj: Any) -> "Category":
        __id = str(obj.get("_id"))
        _name = str(obj.get("name"))
        return Category(__id, _name)


@dataclass
class CreatedBy:
    _id: str
    wallet: str
    username: str
    profileScore: float
    bio: str
    name: str
    profilePicture: str

    @staticmethod
    def from_dict(obj: Any) -> "CreatedBy":
        __id = str(obj.get("_id"))
        _wallet = str(obj.get("wallet"))
        _username = str(obj.get("username"))
        _profileScore = float(obj.get("profileScore"))
        _bio = str(obj.get("bio"))
        _name = str(obj.get("name"))
        _profilePicture = str(obj.get("profilePicture"))
        return CreatedBy(
            __id, _wallet, _username, _profileScore, _bio, _name, _profilePicture
        )


@dataclass
class Day:
    @staticmethod
    def from_dict(obj: Any) -> "Day":
        return Day()


@dataclass
class Owner:
    @staticmethod
    def from_dict(obj: Any) -> "Owner":
        return Owner()


@dataclass
class Operation:
    availableAllTime: bool
    _id: str
    days: List[object]

    @staticmethod
    def from_dict(obj: Any) -> "Operation":
        _availableAllTime = bool(obj.get("availableAllTime"))
        __id = str(obj.get("_id"))
        _days = [Day.from_dict(y) for y in obj.get("days")]
        return Operation(_availableAllTime, __id, _days)


@dataclass
class Reward:
    _id: str
    epoch: int
    product: str
    reviewCycle: int
    __v: int
    createdAt: str
    reward: float
    updatedAt: str

    @staticmethod
    def from_dict(obj: Any) -> "Reward":
        __id = str(obj.get("_id"))
        _epoch = int(obj.get("epoch"))
        _product = str(obj.get("product"))
        _reviewCycle = int(obj.get("reviewCycle"))
        ___v = int(obj.get("__v"))
        _createdAt = str(obj.get("createdAt"))
        _reward = float(obj.get("reward"))
        _updatedAt = str(obj.get("updatedAt"))
        return Reward(
            __id, _epoch, _product, _reviewCycle, ___v, _createdAt, _reward, _updatedAt
        )


from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ReviewedProduct:
    _id: str
    name: str
    currentReviewCycle: int
    category: Optional["Category"]
    description: str
    url: str
    location: str
    operation: Optional["Operation"]
    specialReviewRequest: str
    discountCode: str
    offer: str
    subcategories: List[str]
    slug: str
    gallery: List[object]
    teams: List[object]
    twitterProfile: str
    isClaimed: bool
    isClaiming: bool
    network: str
    createdBy: Optional["CreatedBy"]
    owners: List[object]
    status: str
    reviewDeadline: float
    rewards: List["Reward"]
    createdAt: str
    updatedAt: str
    __v: int
    logo: str
    coverImage: str
    epoch: int
    consensusScore: float
    normalizedTrustScore: float
    trustScore: float
    lastReviewed: str
    ratingScore: float
    reward: float
    id: str
    reviewCount: int
    subscribersCount: int
    isSubscribed: bool
    productCreatorSignature: str
    productCreatorWalletType: str
    publishedAt: str
    managers: List[str]
    adminMessage: str
    isVerified: bool
    listingTier: str
    specialReviewRequestTaskUrl: str

    @staticmethod
    def from_dict(obj: Any) -> "ReviewedProduct":
        def safe_str(val, default=""):
            return str(val) if val is not None else default

        def safe_int(val, default=0):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def safe_bool(val, default=False):
            return bool(val) if val is not None else default

        __id = safe_str(obj.get("_id"))
        _name = safe_str(obj.get("name"))
        _currentReviewCycle = safe_int(obj.get("currentReviewCycle"))
        _category = (
            Category.from_dict(obj.get("category")) if obj.get("category") else None
        )
        _description = safe_str(obj.get("description"))
        _url = safe_str(obj.get("url"))
        _location = safe_str(obj.get("location"))
        _operation = (
            Operation.from_dict(obj.get("operation")) if obj.get("operation") else None
        )
        _specialReviewRequest = safe_str(obj.get("specialReviewRequest"))
        _discountCode = safe_str(obj.get("discountCode"))
        _offer = safe_str(obj.get("offer"))
        _subcategories = [safe_str(y) for y in obj.get("subcategories", [])]
        _slug = safe_str(obj.get("slug"))
        _gallery = [safe_str(y) for y in obj.get("gallery", [])]
        _teams = obj.get("teams", [])
        _twitterProfile = safe_str(obj.get("twitterProfile"))
        _isClaimed = safe_bool(obj.get("isClaimed"))
        _isClaiming = safe_bool(obj.get("isClaiming"))
        _network = safe_str(obj.get("network"))
        _createdBy = (
            CreatedBy.from_dict(obj.get("createdBy")) if obj.get("createdBy") else None
        )
        _owners = [Owner.from_dict(y) for y in obj.get("owners", [])]
        _status = safe_str(obj.get("status"))
        _reviewDeadline = safe_float(obj.get("reviewDeadline"))
        _rewards = [Reward.from_dict(y) for y in obj.get("rewards", [])]
        _createdAt = safe_str(obj.get("createdAt"))
        _updatedAt = safe_str(obj.get("updatedAt"))
        ___v = safe_int(obj.get("__v"))
        _logo = safe_str(obj.get("logo"))
        _coverImage = safe_str(obj.get("coverImage"))
        _epoch = safe_int(obj.get("epoch"))
        _consensusScore = safe_float(obj.get("consensusScore"))
        _normalizedTrustScore = safe_float(obj.get("normalizedTrustScore"))
        _trustScore = safe_float(obj.get("trustScore"))
        _lastReviewed = safe_str(obj.get("lastReviewed"))
        _ratingScore = safe_float(obj.get("ratingScore"))
        _reward = safe_float(obj.get("reward"))
        _id = safe_str(obj.get("id"))
        _reviewCount = safe_int(obj.get("reviewCount"))
        _subscribersCount = safe_int(obj.get("subscribersCount"))
        _isSubscribed = safe_bool(obj.get("isSubscribed"))
        _productCreatorSignature = safe_str(obj.get("productCreatorSignature"))
        _productCreatorWalletType = safe_str(obj.get("productCreatorWalletType"))
        _publishedAt = safe_str(obj.get("publishedAt"))
        _managers = [safe_str(y) for y in obj.get("managers", [])]
        _adminMessage = safe_str(obj.get("adminMessage"))
        _isVerified = safe_bool(obj.get("isVerified"))
        _listingTier = safe_str(obj.get("listingTier"))
        _specialReviewRequestTaskUrl = safe_str(obj.get("specialReviewRequestTaskUrl"))

        return ReviewedProduct(
            __id,
            _name,
            _currentReviewCycle,
            _category,
            _description,
            _url,
            _location,
            _operation,
            _specialReviewRequest,
            _discountCode,
            _offer,
            _subcategories,
            _slug,
            _gallery,
            _teams,
            _twitterProfile,
            _isClaimed,
            _isClaiming,
            _network,
            _createdBy,
            _owners,
            _status,
            _reviewDeadline,
            _rewards,
            _createdAt,
            _updatedAt,
            ___v,
            _logo,
            _coverImage,
            _epoch,
            _consensusScore,
            _normalizedTrustScore,
            _trustScore,
            _lastReviewed,
            _ratingScore,
            _reward,
            _id,
            _reviewCount,
            _subscribersCount,
            _isSubscribed,
            _productCreatorSignature,
            _productCreatorWalletType,
            _publishedAt,
            _managers,
            _adminMessage,
            _isVerified,
            _listingTier,
            _specialReviewRequestTaskUrl,
        )


@dataclass
class ReviewedData:
    products: List[ReviewedProduct]

    @staticmethod
    def from_dict(obj: Any) -> "ReviewedData":
        _products = [ReviewedProduct.from_dict(y) for y in obj.get("products")]
        return ReviewedData(_products)


@dataclass
class ReviewedProductsApiResponse:
    message: str
    data: ReviewedData

    @staticmethod
    def from_dict(obj: Any) -> "ReviewedProductsApiResponse":
        _message = str(obj.get("message"))
        _data = ReviewedData.from_dict(obj.get("data"))
        return ReviewedProductsApiResponse(_message, _data)


@dataclass
class ReviewedProductApiResponse:
    message: str
    data: ReviewedProduct

    @staticmethod
    def from_dict(obj: Any) -> "ReviewedProductApiResponse":
        _message = str(obj.get("message"))
        _data = ReviewedProduct.from_dict(obj.get("data"))
        return ReviewedProductApiResponse(_message, _data)


@dataclass
class UnreviewedProduct:
    _id: str
    name: str
    currentReviewCycle: int
    category: Optional["Category"]
    description: str
    url: str
    location: str
    operation: Optional["Operation"]
    specialReviewRequest: str
    discountCode: str
    offer: str
    subcategories: List[str]
    slug: str
    gallery: List[object]
    teams: List[object]
    twitterProfile: str
    isClaimed: bool
    isClaiming: bool
    network: str
    createdBy: Optional["CreatedBy"]
    owners: List[object]
    status: str
    reviewDeadline: float
    rewards: List["Reward"]
    createdAt: str
    updatedAt: str
    __v: int
    logo: str
    coverImage: str
    epoch: int
    reward: float
    id: str
    subscribersCount: int
    isSubscribed: bool
    consensusScore: float
    lastReviewed: str
    ratingScore: float
    productCreatorSignature: str
    productCreatorWalletType: str
    publishedAt: str
    managers: List[str]
    adminMessage: str
    isVerified: bool
    listingTier: str
    specialReviewRequestTaskUrl: str

    @staticmethod
    def from_dict(obj: Any) -> "UnreviewedProduct":
        def safe_str(val, default=""):
            return str(val) if val is not None else default

        def safe_int(val, default=0):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def safe_bool(val, default=False):
            return bool(val) if val is not None else default

        __id = safe_str(obj.get("_id"))
        _name = safe_str(obj.get("name"))
        _currentReviewCycle = safe_int(obj.get("currentReviewCycle"))
        _category = (
            Category.from_dict(obj.get("category")) if obj.get("category") else None
        )
        _description = safe_str(obj.get("description"))
        _url = safe_str(obj.get("url"))
        _location = safe_str(obj.get("location"))
        _operation = (
            Operation.from_dict(obj.get("operation")) if obj.get("operation") else None
        )
        _specialReviewRequest = safe_str(obj.get("specialReviewRequest"))
        _discountCode = safe_str(obj.get("discountCode"))
        _offer = safe_str(obj.get("offer"))
        _subcategories = [safe_str(y) for y in obj.get("subcategories", [])]
        _slug = safe_str(obj.get("slug"))
        _gallery = [safe_str(y) for y in obj.get("gallery", [])]
        _teams = obj.get("teams", [])
        _twitterProfile = safe_str(obj.get("twitterProfile"))
        _isClaimed = safe_bool(obj.get("isClaimed"))
        _isClaiming = safe_bool(obj.get("isClaiming"))
        _network = safe_str(obj.get("network"))
        _createdBy = (
            CreatedBy.from_dict(obj.get("createdBy")) if obj.get("createdBy") else None
        )
        _owners = [Owner.from_dict(y) for y in obj.get("owners", [])]
        _status = safe_str(obj.get("status"))
        _reviewDeadline = safe_float(obj.get("reviewDeadline"))
        _rewards = [Reward.from_dict(y) for y in obj.get("rewards", [])]
        _createdAt = safe_str(obj.get("createdAt"))
        _updatedAt = safe_str(obj.get("updatedAt"))
        ___v = safe_int(obj.get("__v"))
        _logo = safe_str(obj.get("logo"))
        _coverImage = safe_str(obj.get("coverImage"))
        _epoch = safe_int(obj.get("epoch"))
        _reward = safe_float(obj.get("reward"))
        _id = safe_str(obj.get("id"))
        _subscribersCount = safe_int(obj.get("subscribersCount"))
        _isSubscribed = safe_bool(obj.get("isSubscribed"))
        _consensusScore = safe_float(obj.get("consensusScore"))
        _lastReviewed = safe_str(obj.get("lastReviewed"))
        _ratingScore = safe_float(obj.get("ratingScore"))
        _productCreatorSignature = safe_str(obj.get("productCreatorSignature"))
        _productCreatorWalletType = safe_str(obj.get("productCreatorWalletType"))
        _publishedAt = safe_str(obj.get("publishedAt"))
        _managers = [safe_str(y) for y in obj.get("managers", [])]
        _adminMessage = safe_str(obj.get("adminMessage"))
        _isVerified = safe_bool(obj.get("isVerified"))
        _listingTier = safe_str(obj.get("listingTier"))
        _specialReviewRequestTaskUrl = safe_str(obj.get("specialReviewRequestTaskUrl"))

        return UnreviewedProduct(
            __id,
            _name,
            _currentReviewCycle,
            _category,
            _description,
            _url,
            _location,
            _operation,
            _specialReviewRequest,
            _discountCode,
            _offer,
            _subcategories,
            _slug,
            _gallery,
            _teams,
            _twitterProfile,
            _isClaimed,
            _isClaiming,
            _network,
            _createdBy,
            _owners,
            _status,
            _reviewDeadline,
            _rewards,
            _createdAt,
            _updatedAt,
            ___v,
            _logo,
            _coverImage,
            _epoch,
            _reward,
            _id,
            _subscribersCount,
            _isSubscribed,
            _consensusScore,
            _lastReviewed,
            _ratingScore,
            _productCreatorSignature,
            _productCreatorWalletType,
            _publishedAt,
            _managers,
            _adminMessage,
            _isVerified,
            _listingTier,
            _specialReviewRequestTaskUrl,
        )


@dataclass
class UnreviewedData:
    products: List[UnreviewedProduct]

    @staticmethod
    def from_dict(obj: Any) -> "UnreviewedData":
        _products = [UnreviewedProduct.from_dict(y) for y in obj.get("products")]
        return UnreviewedData(_products)


@dataclass
class UnreviewedProductsApiResponse:
    message: str
    data: UnreviewedData

    @staticmethod
    def from_dict(obj: Any) -> "UnreviewedProductsApiResponse":
        _message = str(obj.get("message"))
        _data = UnreviewedData.from_dict(obj.get("data"))
        return UnreviewedProductsApiResponse(_message, _data)


@dataclass
class UnreviewedProductApiResponse:
    message: str
    data: UnreviewedProduct

    @staticmethod
    def from_dict(obj: Any) -> "UnreviewedProductApiResponse":
        _message = str(obj.get("message"))
        _data = UnreviewedProduct.from_dict(obj.get("data"))
        return UnreviewedProductApiResponse(_message, _data)


# Example Usage
# jsonstring = json.loads(myjsonstring)
# root = Root.from_dict(jsonstring)
