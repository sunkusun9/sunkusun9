"""
Column selection helper functions for resolve_columns
"""

def ohe_drop_first(columns, org_X):
    """OneHotEncoder로 생성된 컬럼에서 각 원래 변수의 첫 번째 더미 변수를 제거

    Args:
        columns: OneHotEncoding 후 컬럼명 리스트
                 형식: {처리단계명}__{원래변수명}_{카테고리값}
        org_X: 원래 변수명 리스트

    Returns:
        boolean list: True면 해당 컬럼을 선택, False면 제외

    Example:
        >>> columns = ['stage__color_red', 'stage__color_blue', 'stage__color_green',
        ...            'stage__size_S', 'stage__size_M', 'stage__size_L']
        >>> org_X = ['color', 'size']
        >>> ohe_drop_first(columns, org_X)
        [False, True, True, False, True, True]
        # 결과: ['stage__color_blue', 'stage__color_green', 'stage__size_M', 'stage__size_L']
    """
    # 각 원래 변수에 대해 첫 번째 컬럼을 만났는지 추적
    first_seen = {var: False for var in org_X}

    mask = []
    for col in columns:
        # '__' 뒤의 부분 추출
        if '__' not in col:
            mask.append(False)
            continue

        suffix = col.split('__', 1)[1]

        # 어떤 org_X 변수에 속하는지 확인
        matched = False
        for org_var in org_X:
            # suffix가 org_var로 시작하는지 확인 (예: color_red는 color로 시작)
            if suffix.startswith(f"{org_var}_"):
                matched = True
                if not first_seen[org_var]:
                    # 첫 번째 컬럼은 제외 (False)
                    mask.append(False)
                    first_seen[org_var] = True
                else:
                    # 나머지는 포함 (True)
                    mask.append(True)
                break

        # org_X의 어떤 변수에도 속하지 않으면 제외
        if not matched:
            mask.append(False)

    return mask
